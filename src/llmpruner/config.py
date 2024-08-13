from dataclasses import dataclass
from typing import Optional

@dataclass
class PrunerConfig:
    r"""
    Initialize PrunerConfig. 

    Args:
        model_name_or_path (`str`):
            The name or path of the model to prune
        pad_token_id (`int`):
            The id of the padding token.
        new_num_attention_heads (`int`):
            The new number of attention heads after pruning. Set to None to not perform pruning of the attention heads.
        new_intermediate_size (`int`):
            The new intermediate size in the MLP layer of the model after pruning. Set to None to not perform pruning of the intermediate dimension.
        new_num_hidden_size (`int`):
            The new number of embedding channels after pruning. Set to None to not perform pruning of the embedding channels.
        new_num_hidden_layers (`int`):
            The new number of hidden layers after depth pruning. Set to None to not perform pruning of the layers.
    """
    
    model_name_or_path: str 
    pad_token_id: int = None
    new_num_attention_heads: Optional[int] = None
    new_intermediate_size: Optional[int] = None
    new_hidden_size: Optional[int] = None
    new_num_hidden_layers: Optional[int] = None
    
    def __post_init__(self):
        assert any(
            [dimension is not None for dimension in [self.new_num_attention_heads, self.new_intermediate_size, self.new_hidden_size, self.new_num_hidden_layers]]
        ), "Pruning requires at least one dimension to be pruned."