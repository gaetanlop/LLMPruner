from .width_utils import (
    get_model, 
    prune_attention_heads, 
    prune_neurons,
    prune_embeddings
)
from .depth_utils import prune_layers
from .mapping import MLP_LAYERS_TO_PRUNE, ATTN_LAYERS_TO_PRUNE, RMSNORM_LAYERS_TO_PRUNE
from .config import PrunerConfig
import torch.nn as nn
from typing import Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from transformers.utils import logging
import torch.nn.functional as F


logger = logging.get_logger(__name__)


class PrunerModel(nn.Module):
    r"""
    Initialize PrunerModel. It introduces the latest structured pruning methods for LLMs such as depth pruning using block importance
    and width pruning of the heads in the multi-head attention, the neurons in the MLP layers and the embedding channels.

    Args:
        config (`PrunerConfig`):
            The prunning config. Used to specify the type of pruning that needs to be performed.
        kwargs:
            The arguments are passed to instantiate an AutoModelForCausalLM using the from_pretrained method.
    """
    
    
    def __init__(
        self, 
        config: PrunerConfig, 
        **kwargs
    ):
        super().__init__()
        self.config = config
        
        if kwargs.get("attn_implementation", None)=="flash_attention_2":
            logger.warning(
                "Flash attention 2 is not supported by the Pruner."
                "Setting `attn_implementation` to 'sdpa'"
            )
            kwargs["attn_implementation"] = "sdpa"
            
        self.model, self.architecture_name = get_model(config, **kwargs)
        self.num_layers = self.model.config.num_hidden_layers
        
        self.attention_pruning = self.config.new_num_attention_heads is not None
        self.embedding_pruning = self.config.new_hidden_size is not None
        self.depth_pruning = self.config.new_num_hidden_layers is not None
        self.mlp_pruning = self.config.new_intermediate_size is not None
        
        self._register_tensors()
        self._setup_hooks()
        
    def _register_tensors(self):
        self.head_importance = torch.zeros(self.model.config.num_attention_heads, device=self.model.device)
        self.mlp_importance = torch.zeros(self.model.config.intermediate_size, device=self.model.device)
        self.norm_importance = torch.zeros(self.model.config.hidden_size, device=self.model.device)
        self.block_similarity = torch.zeros(self.num_layers, device=self.model.device)
        self.nb_tokens = torch.tensor(0.0, device=self.model.device)
            
    def _setup_hooks(self) -> None:
        self.rms_norm_activations = []
        def rms_norm_hook_fn(module, input, output):
            self.rms_norm_activations.append(output)
            
        self.attn_activations = []
        def attn_hook_fn(module, input, output):
            self.attn_activations.append(output[-1])
            return output[:-1]
            
        self.mlp_activations = []
        def mlp_hook_fn(module, input, output):
            self.mlp_activations.append(output[-1])
            return output[0]
    
        for module in self.model.modules():
            if isinstance(module, ATTN_LAYERS_TO_PRUNE[self.architecture_name]):
                module.register_forward_hook(attn_hook_fn)
                
        for module in self.model.modules():
            if isinstance(module, RMSNORM_LAYERS_TO_PRUNE[self.architecture_name]):
                module.register_forward_hook(rms_norm_hook_fn)
                
        for module in self.model.modules():
            if isinstance(module, MLP_LAYERS_TO_PRUNE[self.architecture_name]):
                module.register_forward_hook(mlp_hook_fn)
        
    def prune(self) -> None:
        
        if self.embedding_pruning:
            prune_embeddings(self.model, self.norm_importance, self.config.new_hidden_size)
        
        if self.mlp_pruning:
            prune_neurons(self.model, self.mlp_importance, self.config.new_intermediate_size)
        
        if self.attention_pruning:
            prune_attention_heads(self.model, self.head_importance, self.config.new_num_attention_heads)
            
        if self.depth_pruning:
            block_importance = 1 - self.block_similarity/self.nb_tokens
            print(block_importance)
            prune_layers(self.model, block_importance, self.config.new_num_hidden_layers)
            
    
    def _get_width_importance(self) -> None:

        if self.config.new_num_attention_heads is not None:
            self.head_importance += torch.stack(self.attn_activations, dim=0).norm(dim=-1).mean(dim=(1,3)).sum(dim=0)
        if self.config.new_intermediate_size is not None:
            self.mlp_importance += torch.stack(self.mlp_activations, dim=0).mean(dim=(1,2)).sum(dim=0)
        if self.config.new_hidden_size is not None:
            self.norm_importance += torch.stack(self.rms_norm_activations, dim=0).mean(dim=(1,2)).sum(dim=0)
    
        self.rms_norm_activations = []
        self.mlp_activations = []
        self.attn_activations = []
        
    def _get_block_importance(self, input_ids, hidden_states) -> None:
        mask = input_ids[:, 1:] != self.config.pad_token_id
        
        for i in range(self.num_layers):
            # We exclude the bos tokens from calculations
            hidden_states_input = hidden_states[i][:, 1:, :]
            hidden_states_output = hidden_states[i + 1][:, 1:, :]
            
            # Avoid nan values in cosine similarity computation
            hidden_states_input[hidden_states_input == 0] = torch.finfo(hidden_states_input.dtype).eps
            hidden_states_output[hidden_states_output == 0] = torch.finfo(hidden_states_output.dtype).eps

            cosine_sim = F.cosine_similarity(hidden_states_input, hidden_states_output, dim=-1)

            # Mask out padding tokens
            masked_cosine_sim = cosine_sim * mask

            self.block_similarity[i] += masked_cosine_sim.sum()
        
        self.nb_tokens += mask.sum()
                
    def forward(
        self, 
        input_ids: torch.Tensor = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        kwargs["output_hidden_states"] = True if self.config.new_num_hidden_layers is not None else False
        outputs = self.model(input_ids, **kwargs)    
        
        if self.depth_pruning:
            self._get_block_importance(input_ids, outputs.hidden_states)
        
        self._get_width_importance()
        
        return outputs
    
    def save_pretrained(self, save_path: str) -> None:
        self.model.save_pretrained(save_path)
        
        
        