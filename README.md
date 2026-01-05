# STrace


```angular2html
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

We provide `get_log_likelihood()` and `generate()` functions in `get_log_likelihood.py` 
and `generate.py` respectively, for conditional likelihood evaluation and conditional generation.





python eval_llada.py \
  --model llada_dist \
  --model_args '{"model_path": "GSAI-ML/LLaDA-8B-Instruct", "mc_num": 8, "is_check_greedy": false, "cfg": 0.0, "device": "cuda"}' \
  --tasks gsm8k \
  --num_fewshot 0 \
  --limit 1
  --verbosity DEBUG \
  --log_samples \
  --output_path llada_gsm8k_results > output2.txt



































