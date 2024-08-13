import torch
import torch.nn as nn

# inspired by prune_linear_layer in https://github.com/huggingface/transformers/blob/06146e312a5e2d588fcb10810c9c06410d1fa078/src/transformers/pytorch_utils.py#L53
def prune_embedding_layer(layer: nn.Embedding, index: torch.LongTensor) -> nn.Linear:
    """
    Prune an embedding layer to keep only entries in index.

    Used to remove embedding channels.

    Args:
        layer (`torch.nn.Embedding`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.

    Returns:
        `torch.nn.Embedding`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(1, index).clone().detach()
    new_size = list(layer.weight.size())
    new_size[1] = len(index)
    new_layer = nn.Embedding(new_size[0], new_size[1], padding_idx=layer.padding_idx).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    return new_layer