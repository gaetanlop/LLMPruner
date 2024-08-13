# LLMPruner

Training large-scale language models with billions of parameters demands significant resources and time. Recently, NVIDIA released a [paper](https://www.arxiv.org/pdf/2407.14679) studying the best practices to compress a LLM using structured pruning. LLMPruner is an unofficial implementation of the pruning recipes developed in the paper. Specifically, LLMPruner offers tools for structured pruning of LLMs using only forward passes (it supports both width and depth pruning), minimizing memory usage and data requirements.

## Quickstart

Install LLMPruner from source.

```bash
git clone ...
cd llmpruner
pip install .
```

LLMPruner supports structured pruning on the width (attentions heads, MLP neurons, embedding channels) and depth (layers pruning) dimensions. 

Performing width pruning on the attention heads and the neurons in the FFN layers.

```python
from llmpruner import PrunerConfig, PrunerModel

config = PrunerConfig(
        model_name_or_path=args.model_name_or_path,
        new_hidden_size=args.new_hidden_size,
        new_intermediate_size=args.new_intermediate_size,
        new_num_attention_heads=args.new_num_attention_heads
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
- [] Add Gemma2 architecture support
- [] Add Mistral architecture support

## Future directions

This library was developed during an 12-hour flight to ACL 2024 in Bangkok, as a way to make productive use of the long time of this flight. I may not have the bandwidth to continuously update it or add new structured pruning methods. 

## Contact

You can contact me on [linkedin](https://www.linkedin.com/in/gaetan-lopez/) if you have any feature requests or want some help to setup the library.