# Small LLaDA variants for PCFG training

Size knobs for 100M / 200M / 300M live in one file: `variants.json`.
Tokenizer is shared in this package (same vocab for all sizes).

| Variant | Shape | ~Params |
|---------|-------|---------|
| `100m` | d=768, L=12, H=12, mlp=2048 | 85.2M |
| `200m` | d=1024, L=16, H=16, mlp=3072 | 218.4M |
| `300m` | d=1024, L=24, H=16, mlp=3072 | 327.5M |

Shared tokenizer: `<pad>=0`, `<mask>=1`, `<eos>=2`, then grammar terminals (`vocab_size=51`,
`model_max_length=128`).

## Load

```python
from models.pcfg_models import load_pcfg_llada, load_pcfg_tokenizer

model, config, vocab = load_pcfg_llada("100m")  # or "200m" / "300m"
tokenizer = load_pcfg_tokenizer()
```

## Train

```bash
python llada-pcfg/scripts/train_pcfg_llada.py --variant 100m --hf-dataset pcfg-code --steps 50
```
