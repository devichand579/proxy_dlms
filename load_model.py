"""
Load LLaDA model  without using HuggingFace's from_pretrained mechanism.
"""

import torch
import json
from pathlib import Path
import sys
from models.llada.configuration_llada import LLaDAConfig
from models.llada.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer



#  models directory to path
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models" / "llada"
sys.path.insert(0, str(BASE_DIR))


def load_config(model_dir: str):
    """
    Load model configuration from local config.json file.
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object from the dictionary
    config = LLaDAConfig(**config_dict)
    return config


def load_model(model_dir: str, torch_dtype=torch.bfloat16, device="cuda"):
    """
    Load LLaDA model completely from local files.
    """
    model_dir = Path(model_dir)
    
    # Load config
    config = load_config(model_dir)
    
    # Load model using HuggingFace's from_pretrained but with local path
    # This still uses HF's safetensors loading but from local files
    model = LLaDAModelLM.from_pretrained(
        str(model_dir),
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True 
    )
    
    model = model.to(device)
    model.eval()
    
    return model


def load_tokenizer(model_dir: str):
    """
    Load tokenizer from local files.
    """
    model_dir = Path(model_dir)
    
    # Check if tokenizer files exist
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    for file in tokenizer_files:
        if not (model_dir / file).exists():
            raise FileNotFoundError(f"Tokenizer file {file} not found at {model_dir}")
    
    # Load tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False  # Not needed since we're loading locally
    )
    
    return tokenizer


def load_model_and_tokenizer(model_dir: str, torch_dtype=torch.bfloat16, device="cuda"):
    """
    function to load both model and tokenizer from local files.
    """
    model = load_model(model_dir, torch_dtype=torch_dtype, device=device)
    tokenizer = load_tokenizer(model_dir)
    
    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    model_dir = "./models/llada"
    
    print(f"Loading model from {model_dir}...")
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    print("Model and tokenizer loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Tokenizer type: {type(tokenizer)}")

