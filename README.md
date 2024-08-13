# LLMPruner

Training large-scale language models (LLMs) with billions of parameters requires immense computational resources and time. In a [recent paper](https://www.arxiv.org/pdf/2407.14679), NVIDIA explored best practices for compressing LLMs through structured pruning. LLMPruner is an unofficial implementation of the pruning strategies presented in that paper. The library provides tools for structured pruning of LLMs using only forward passes, enabling both width and depth pruning while minimizing memory usage.

## Quickstart

Install LLMPruner from source.

```bash
git clone https://github.com/gaetanlop/LLMPruner.git
cd llmpruner
pip install .
```

LLMPruner supports structured pruning on the width (attentions heads, MLP neurons, embedding channels) and depth (layers pruning) dimensions. 

Example of performing width pruning on the attention heads, the neurons in the FFN layers and the embedding channels:

```python
from llmpruner import PrunerConfig, PrunerModel

config = PrunerConfig(
        model_name_or_path=model_name_or_path,
        new_hidden_size=new_hidden_size,
        new_intermediate_size=new_intermediate_size,
        new_num_attention_heads=new_num_attention_heads
    )

# The pruner model takes in a PrunerConfig and any arguments you would have used to instantiate a transformer model with from_pretrained such as the dtype, the device and the attention implementation.
model = PrunerModel(config, torch_dtype="float16")

# Make forward passes over your data
for batch in dataloader:
    with torch.inference_mode():
        _ = model(batch["input_ids"])

model.prune()
model.save_pretrained(pruned_model_save_path)
```

You can load the pruned model for re-training or inference using transformers.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(pruned_model_save_path)
```

## Model Support

| Model   | Is Supported |
|---------|--------------|
| Gemma2  | No (soon)    |
| LLaMA   | Yes          |
| Mistral | No (soon)    |

More architectures might be added in the near future.

## Task list

- [X] Add Llama architecture support
- [ ] Add Gemma2 architecture support
- [ ] Add Mistral architecture support

## Future directions

This library was developed during an 12-hour flight to ACL 2024 in Bangkok. I may not have the bandwidth to continuously update it or add new structured pruning methods. 

## Contact

You can contact me on [linkedin](https://www.linkedin.com/in/gaetan-lopez/) if you have any feature requests or want some help to setup the library.