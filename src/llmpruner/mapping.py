from models.llama import LlamaAttention, LlamaMLP, LlamaRMSNorm, LlamaForCausalLM

SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM": LlamaForCausalLM,
}

MLP_LAYERS_TO_PRUNE = {
    "LlamaForCausalLM": LlamaMLP,
}

ATTN_LAYERS_TO_PRUNE = {
    "LlamaForCausalLM": LlamaAttention,
}

RMSNORM_LAYERS_TO_PRUNE = {
    "LlamaForCausalLM": LlamaRMSNorm,
}