import torch
from transformers import AutoConfig, AutoModelForCausalLM
from .mapping import SUPPORTED_ARCHITECTURES
from typing import Tuple

      
def prune_attention_heads(
    model: AutoModelForCausalLM, 
    head_importance: torch.FloatTensor, 
    new_num_attention_heads: int
) -> None:  
    """
    Prunes the attention heads of a transformer-based language model based on their importance scores.

    Args:
        model (`PrunerModel`): 
            The PrunerModel containing the model to prune.
        head_importance (`torch.FloatTensor`): 
            A tensor containing importance scores for each attention head in the model.
        new_num_attention_heads (`int`): 
            The number of attention heads to keep after pruning. 

    Returns:
        None: The function modifies the model inplace. 
    """
    num_attention_heads_to_prune =  len(head_importance) - new_num_attention_heads
    _, heads_to_prune = torch.topk(head_importance, num_attention_heads_to_prune, largest=False)
    model.prune_heads(heads_to_prune)

def prune_neurons(
    model: AutoModelForCausalLM, 
    mlp_importance: torch.FloatTensor, 
    new_intermediate_size: int
) -> None:
    """
    Prunes the intermediate size in the MLP layers of a transformer-based language model based on their importance scores.

    Args:
        model (`PrunerModel`): 
            The PrunerModel containing the model to prune.
        mlp_importance (`torch.FloatTensor`): 
            A tensor containing importance scores for each neurons in the linear layers of the MLP layers in the model.
        new_intermediate_size (`int`): 
            The number of neurons to keep after pruning in the intermediade size of the MLP model.. 

    Returns:
        None: The function modifies the model inplace. 
    """
    _, neurons_to_keep = torch.topk(mlp_importance, new_intermediate_size)
    model.prune_neurons(neurons_to_keep)
    
def prune_embeddings(
    model: AutoModelForCausalLM, 
    emb_importance: torch.FloatTensor, 
    new_embedding_size: int
) -> None:
    """
    Prunes the embedding channel of a transformer-based language model based on their importance scores.

    Args:
        model (`PrunerModel`): 
            The PrunerModel containing the model to prune.
        emb_importance (`torch.FloatTensor`): 
            A tensor containing importance scores for each neurons in the embedding channel dimension.
        new_embedding_size (`int`): 
            The number of neurons to keep after pruning in the embedding channel. 

    Returns:
        None: The function modifies the model inplace. 
    """
    _, emb_to_keep = torch.topk(emb_importance, new_embedding_size)    
    model.prune_embeddings(emb_to_keep)
    
   
def get_model(pruner_config, **kwargs) -> Tuple[AutoModelForCausalLM, str]:
    """
    Loads a PrunerModel.

    Returns:
        Tuple[AutoModelForCausalLM, str]: A tuple containing the loaded model and the architecture type as a string.
    """
    
    config = AutoConfig.from_pretrained(pruner_config.model_name_or_path)
    
    if pruner_config.new_hidden_size is not None:
        assert config.hidden_size>pruner_config.new_hidden_size, f"Pruning requires `new_hidden_size` to be smaller than the `hidden_size`of the current model. Currently, hidden_size={config.hidden_size} and new_hidden_size={pruner_config.new_hidden_size}"
            
    if pruner_config.new_intermediate_size is not None:
        assert config.intermediate_size>pruner_config.new_intermediate_size, f"Pruning requires `new_intermediate_size` to be smaller than the `intermediate_size`of the current model. Currently, intermediate_size={config.intermediate_size} and new_intermediate_size={pruner_config.intermediate_size}"
            
    if pruner_config.new_num_attention_heads is not None:
        assert config.num_attention_heads>pruner_config.new_num_attention_heads, f"Pruning requires `new_num_attention_heads` to be smaller than the `num_attention_heads`of the current model. Currently, num_attention_heads={config.num_attention_heads} and new_num_attention_heads={pruner_config.new_num_attention_heads}"
    
    
    if config.architectures[0] not in list(SUPPORTED_ARCHITECTURES):
        raise ValueError(
            f"{config.architectures[0]} not supported."
            f"Currently supported architectures: {list(SUPPORTED_ARCHITECTURES)}"  
        )
    
    model = SUPPORTED_ARCHITECTURES[config.architectures[0]].from_pretrained(pruner_config.model_name_or_path, **kwargs)
    
    return model, config.architectures[0]