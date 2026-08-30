import hydra
from omegaconf import OmegaConf, DictConfig
cfg = OmegaConf.create({
    "model_path": "./models/llada",
    "eval": {
        "model_path": "${model_path}"
    }
})
base = OmegaConf.create({"hidden_dir": "outputs/eval_hidden", "checkpoint": "x", "model_path": None})
o = cfg.get("eval")
resolved = OmegaConf.to_container(o, resolve=True)
ev = OmegaConf.merge(base, OmegaConf.create(resolved))
print(ev.get("model_path"))
