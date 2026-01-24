# STrace

## Usage

### Environment Setup
Run the following command to create a new environment and install the dependencies.
```
conda create -n strace -f environment.yaml
conda activate strace
pip install -r requirements.txt
```
### Downloading the model configuration 
Please download the model configuration and safetensors files before running any scheduler or evaluation scripts.

```
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='GSAI-ML/LLaDA-8B-Base', local_dir='./models/llada', local_dir_use_symlinks=False)"
```




python eval_llada.py \
  --model llada_dist \
  --model_args '{"model_path": "GSAI-ML/LLaDA-8B-Instruct", "mc_num": 8, "is_check_greedy": false, "cfg": 0.0, "device": "cuda"}' \
  --tasks gsm8k \
  --num_fewshot 0 \
  --limit 1
  --verbosity DEBUG \
  --log_samples \
  --output_path llada_gsm8k_results > output2.txt

We provide `get_log_likelihood()` and `generate()` functions in `get_log_likelihood.py` 
and `generate.py` respectively, for conditional likelihood evaluation and conditional generation.
































