from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from llmpruner import PrunerConfig, PrunerModel
import torch
from torch.utils.data import DataLoader
import argparse
from datasets import load_dataset
import itertools


def main(args):
    
    # Change the calibration dataset to fit your needs.
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    config = PrunerConfig(
        model_name_or_path=args.model_name_or_path,
        pad_token_id=tokenizer.pad_token_id,
        new_hidden_size=args.new_hidden_size,
        new_intermediate_size=args.new_intermediate_size,
        new_num_attention_heads=args.new_num_attention_heads,
        new_num_hidden_layers=args.new_num_hidden_layers
    )
    model = PrunerModel(config, torch_dtype=args.torch_dtype)
    model.eval()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
        
    )
    
    def preprocess_and_tokenize(example):
        model_inputs = tokenizer(example["text"], max_length=args.max_length, truncation=True, add_special_tokens=True) 
        return model_inputs
    
    dataset = dataset.map(preprocess_and_tokenize, batched=True, remove_columns=dataset.column_names)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False
    )
    
    if args.max_num_iterations!=-1:
        dataloader = itertools.islice(dataloader, args.max_num_iterations)

    for batch in dataloader:
        with torch.inference_mode():
            _ = model(batch["input_ids"])

    print(model)
    model.prune()
    print(model)

    # outputs = model(batch["input_ids"])
    
    print("Saving the model...")
    model.save_pretrained(args.save_path)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--new_hidden_size", type=int, required=False)
    parser.add_argument("--new_intermediate_size", type=int, required=False)
    parser.add_argument("--new_num_attention_heads", type=int, required=False)
    parser.add_argument("--new_num_hidden_layers", type=int, required=False)
    
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, required=True, default=32, help="The maximum length of each sample.")
    parser.add_argument("--save_path", type=str, required=True)
    
    parser.add_argument("--max_num_iterations", type=int, default=-1)
    args = parser.parse_args()
    main(args)




