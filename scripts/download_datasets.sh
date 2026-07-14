#!/bin/bash
# Script to download datasets for offline use on compute nodes
# (lm-eval benchmarks + PCFG training corpora pcfg-code, pcfg-code-1B)

# shellcheck source=_repo_root.sh
_script_dir="$(dirname "${BASH_SOURCE[0]}")"
# When sbatch copies the script to its spool dir, _repo_root.sh won't be there;
# fall back to the original submission directory.
if [[ ! -f "$_script_dir/_repo_root.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _script_dir="${SLURM_SUBMIT_DIR}/scripts"
fi
source "$_script_dir/_repo_root.sh"
unset _script_dir

# Set cache directories
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export HF_HOME="$HOME/.cache/huggingface"

# Set HuggingFace token for authentication (required for some datasets)
export HF_TOKEN="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"  # NOTE: DON'T COMMIT THIS TOKEN

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

# Each entry: (full_org_name, short_name_alias, config, display_name)
# Both names are cached so offline mode works regardless of which name lm-eval uses.
datasets_to_download = [
    ("openai/gsm8k",                    "gsm8k",            "main",          "GSM8K"),
    ("google-research-datasets/mbpp",   "mbpp",             "full",          "MBPP"),
    ("BAAI/TACO",                        None,               "ALL",           "TACO"),
    ("openai/openai_humaneval",          "openai_humaneval", None,            "HumanEval"),
    ("allenai/ai2_arc",                  None,               "ARC-Challenge", "ARC Challenge"),
    ("HuggingFaceH4/MATH-500",           None,               None,            "Minerva MATH-500"),
]

print("\n" + "="*60)
for full_name, short_name, config_name, display_name in datasets_to_download:
    names_to_cache = [n for n in [full_name, short_name] if n]
    print(f"\nDownloading {display_name} dataset...")
    for name in names_to_cache:
        try:
            if config_name:
                dataset = load_dataset(name, config_name, token=hf_token)
            else:
                dataset = load_dataset(name, token=hf_token)
            print(f"  Cached as '{name}' ({display_name})")
            for split_name in dataset.keys():
                print(f"   - {split_name}: {len(dataset[split_name])} samples")
        except Exception as e:
            print(f"  Warning: could not cache as '{name}': {e}")
            continue

print("\n" + "="*60)
print("Downloading PCFG training corpora...")
try:
    import sys
    from pathlib import Path

    repo_root = Path(os.getcwd())
    sys.path.insert(0, str(repo_root / "llada-pcfg"))
    from hf_hub import load_pcfg_dataset, resolve_pcfg_repo

    pcfg_datasets = ["pcfg-code", "pcfg-code-1B"]
    for dataset_name in pcfg_datasets:
        print(f"\n  {dataset_name}...")
        try:
            full_repo = resolve_pcfg_repo(dataset=dataset_name)
            pcfg_ds = load_pcfg_dataset(dataset=dataset_name, token=hf_token)
            print(f"  Cached as '{full_repo}'")
            for split_name in pcfg_ds.keys():
                print(f"   - {split_name}: {len(pcfg_ds[split_name])} samples")
        except Exception as e:
            print(f"  Warning: could not cache '{dataset_name}': {e}")
except Exception as e:
    print(f"  Warning: could not load PCFG helpers: {e}")

print("\n" + "="*60)
print(f"All datasets downloaded to: {os.environ.get('HF_DATASETS_CACHE')}")
PYTHON_EOF

echo ""
echo "================================================"
echo "Done! Datasets are cached and ready for offline use."
