#!/usr/bin/env python3
"""
Download and tokenize TinyStoriesV2-GPT4 files
"""

import os
import argparse
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=os.path.basename(filename),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Download and tokenize TinyStories dataset")
    parser.add_argument("data_dir", help="Directory to download files and save tokenized data")
    parser.add_argument("--tokenizer-model", default="roneneldan/TinyStories-1M")
    
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Download files
    base_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    files = ["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"]
    
    for filename in files:
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            download_file(base_url + filename, filepath)
            print(f"✓ Downloaded {filename}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    
    # Process files and create datasets
    datasets = {}
    for filename in files:
        split_name = "train" if "train" in filename else "validation"
        filepath = os.path.join(args.data_dir, filename)
        
        print(f"Processing {split_name}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split by <|endoftext|> and create dataset
        stories = [story.strip() for story in text.split('<|endoftext|>') if story.strip()]
        ds = Dataset.from_dict({"text": stories})
        
        # Tokenize
        ds = ds.map(lambda x: {"tokens": tokenizer(x["text"], add_special_tokens=False)["input_ids"]}, batched=True)
        ds = ds.map(lambda x: {"tokens_str": tokenizer.convert_ids_to_tokens(x["tokens"])})
        datasets[split_name] = ds
        
        print(f"✓ {split_name}: {len(stories)} stories")
    
    # Save as HuggingFace dataset
    dataset_dict = DatasetDict(datasets)
    output_path = os.path.join(args.data_dir, "tokenized_tinystories")
    dataset_dict.save_to_disk(output_path)
    print(f"✓ Saved dataset to {output_path}")


if __name__ == "__main__":
    main() 