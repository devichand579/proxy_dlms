#!/bin/bash
# Script to download datasets for offline use on compute nodes

# Set cache directories
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export HF_HOME="$HOME/.cache/huggingface"

# Set HuggingFace token for authentication (required for some datasets)
export HF_TOKEN="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"

echo "Downloading datasets to: $HF_DATASETS_CACHE"
echo "================================================"

python << 'PYTHON_EOF'
from datasets import load_dataset
from huggingface_hub import login
import os

# Login with HF token
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    try:
        login(token=hf_token)
        print("Authenticated with HuggingFace Hub")
    except Exception as e:
        print(f"Warning: Could not authenticate: {e}")

datasets_to_download = [
    ("openai/gsm8k", "main", "GSM8K"),
    ("google-research-datasets/mbpp", "full", "MBPP"),
    ("BAAI/TACO", "ALL", "TACO"),
    ("openai/openai_humaneval", None, "HumanEval"),
    ("allenai/ai2_arc", "ARC-Challenge", "ARC Challenge"),
    ("HuggingFaceH4/MATH-500", None, "Minerva MATH-500"),
]

print("\n" + "="*60)
for dataset_name, config_name, display_name in datasets_to_download:
    print(f"\nDownloading{display_name} dataset...")
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, token=hf_token)
        else:
            dataset = load_dataset(dataset_name, token=hf_token)
        
        print(f"{display_name} downloaded successfully!")
        
        # Show split information
        for split_name in dataset.keys():
            print(f"   - {split_name}: {len(dataset[split_name])} samples")
    except Exception as e:
        print(f"Error downloading {display_name}: {e}")
        print(f"   Dataset: {dataset_name}, Config: {config_name}")
        continue

print("\n" + "="*60)
print(f"All datasets downloaded to: {os.environ.get('HF_DATASETS_CACHE')}")
PYTHON_EOF

echo ""
echo "================================================"
echo "Done! Datasets are cached and ready for offline use."
