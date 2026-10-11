"""Evidence-guided batches of latent scheduler configurations, with one model load."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from schedulerlens.latent_bench import run


PLANS = {
    "eb_latent": [
        {"variants": "eb_gradient_gain", "gain_beta": .25},
        {"variants": "eb_gradient_gain", "gain_beta": .5},
        {"variants": "eb_temporal,eb_target_temporal", "stability_beta": .15},
        {"variants": "eb_target_temporal", "stability_beta": .05},
    ],
    "eb": [
        {"variants": "eb_sampler"},
        {"variants": "eb_sampler", "block_length": 64},
        {"variants": "eb_sampler", "gen_length": 256, "block_length": 256},
    ],
    "second": [
        {"variants": "confidence,gradient_gain", "block_length": 64},
        {"variants": "response_gain", "probe_every": 32},
        {"variants": "confidence,gradient_gain", "gen_length": 256, "block_length": 256},
        {"variants": "gradient_gain", "gain_beta": .5},
    ],
    "gradient": [
        {"variants": "gradient_gain", "gain_beta": 1.},
        {"variants": "gradient_gain", "gain_beta": .75},
        {"variants": "gradient_gain", "gain_beta": .5, "probe_every": 4},
        {"variants": "confidence,gradient_gain", "block_length": 128, "gain_beta": .5},
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=tuple(PLANS), default="second")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    model_bundle = None
    for overrides in PLANS[args.plan]:
        config = dict(model=Path("models/llada"), dataset_arrow=None, samples=args.samples,
                      steps=128, gen_length=512, block_length=512, batch_size=args.batch_size,
                      seed=1234, variants="confidence", diversity_beta=.25, stability_beta=.15,
                      probe_every=8, fisher_vocab=128, gain_beta=.25, response_incoming_beta=.2)
        config.update(overrides)
        suffix = "" if config["gen_length"] == 512 else f"_g{config['gen_length']}"
        config["out"] = Path(f"outputs/latent_humaneval20{suffix}")
        model_bundle = run(SimpleNamespace(**config), model_bundle=model_bundle)


if __name__ == "__main__":
    main()
