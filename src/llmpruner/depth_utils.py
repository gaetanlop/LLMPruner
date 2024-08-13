 
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import List
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F


def compute_perplexity(model: AutoModelForCausalLM, dataloader: DataLoader, max_number_of_iterations: int) -> torch.FloatTensor:
    """
    Computes perplexity of a model on a particular dataloader.

    Args:
        model (`AutoModelForCausalLM`): 
            The model to evaluate.
        dataloader (`DataLoader`): 
            The dataloader that contains samples to evaluate the perplexity of the model.
        max_number_of_iterations (`int`): 
            The maximum number of iterations or batches to process from the dataloader. 

    Returns:
        `torch.FloatTensor`:
            A float tensor containing the perplexity result
    """
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in itertools.islice(dataloader, max_number_of_iterations):
        nb_tokens = torch.sum(batch["labels"]!=-100)
        batch = {k:v.to(model.device) for k,v in batch.items()}
        loss = model(**batch, use_cache=False).loss
        total_loss += loss.item() * nb_tokens
        total_tokens += nb_tokens
        
    return torch.exp(total_loss / total_tokens)


@torch.inference_mode
def ppl_based_ranking(model: AutoModelForCausalLM, dataloader: DataLoader, max_number_of_iterations: int) -> List[torch.FloatTensor]:
    """
    Computes the importance of each layer of an LLM using perplexity as the evaluation criteria.

    Args:
        model (`AutoModelForCausalLM`): 
            The model to evaluate.
        dataloader (`DataLoader`): 
            The dataloader that contains samples to evaluate the perplexity of the model.
        max_number_of_iterations (`int`): 
            The maximum number of iterations or batches to process from the dataloader. 
    Returns:
        `List[torch.FloatTensor]`: 
            A list containing the perplexity values computed after removing each layer. 
    """
    
    print("Computing importance of each layer using perplexity...")
    perplexity = []
    for layer_idx in tqdm(range(len(model.model.layers))):
        original_layers = model.model.layers
        model.model.layers = nn.ModuleList(
            [layer for i, layer in enumerate(original_layers) if i != layer_idx]
        )
        ppl = compute_perplexity(model, dataloader, max_number_of_iterations)
        perplexity.append(ppl)
        model.model.layers = original_layers  # restore original layers
        
    return perplexity


def prune_layers(model: AutoModelForCausalLM, importance: torch.FloatTensor, new_num_hidden_layers: int):
    """
    Prunes the layers of an LLM based on their importance scores.

    Args:
        model (`AutoModelForCausalLM`): 
            The model to prune
        importance (`torch.FloatTensor`): 
            A tensor containing importance scores for each layer in the model.
        new_num_hidden_layers (`int`): 
            The number of layers to keep.

    Returns:
        `AutoModelForCausalLM`: 
            The pruned model with only the most important layers retained. 
    """
    
    _, layers_to_keep = torch.topk(importance, new_num_hidden_layers, dim=0)
    
    model.model.layers = nn.ModuleList(
            [layer for i, layer in enumerate(model.model.layers) if i in layers_to_keep]
        )
    
    # change config for later loading
    model.config.num_hidden_layers = len(layers_to_keep)
    
    return model


            