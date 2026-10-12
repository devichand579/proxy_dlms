import math
import unittest

import numpy as np

from schedulerlens.proxies import (
    TraceProxyError,
    compute_sample_proxies,
    trajectory_blocks,
)


def _synthetic_trace() -> dict:
    gen_length = 8
    selections = ((0, 1), (2,), (3, 4), (5,), (6,), (7,))
    confidences = np.ones(gen_length, dtype=np.float64)
    confidences[[0, 1, 2, 3, 4]] = [0.5, 0.0, 2.0, 0.25, 0.25]
    entropies = np.full(gen_length, 2.0, dtype=np.float64)
    entropies[[0, 1, 2, 3, 4]] = [0.2, 0.7, 1.0, 1.1, 0.9]

    mask = np.ones(gen_length, dtype=bool)
    steps = []
    for positions in selections:
        selected = np.zeros(gen_length, dtype=bool)
        selected[list(positions)] = True
        selected_tokens = np.full(gen_length, -1, dtype=np.int64)
        selected_tokens[list(positions)] = np.arange(
            10, 10 + len(positions), dtype=np.int64
        )
        steps.append(
            {
                "mask_index": mask.copy(),
                "predictions": np.arange(gen_length, dtype=np.int64),
                "confidences": confidences.copy(),
                "entropies": entropies.copy(),
                "selected_positions": selected,
                "selected_tokens": selected_tokens,
            }
        )
        mask[selected] = False

    return {
        "gen_length": gen_length,
        "schedulerlens_trimmed_tail": True,
        "step_traces": steps,
    }


class ProxyTests(unittest.TestCase):
    def test_proxy_formulas_and_early_gating(self) -> None:
        result = compute_sample_proxies(_synthetic_trace())

        expected_m = -math.log(0.5) - math.log(1e-8) - math.log(0.25)
        self.assertAlmostEqual(result.M_hat, expected_m)
        self.assertAlmostEqual(result.J_hat_early, 0.2)
        self.assertAlmostEqual(result.G_hat, expected_m + 0.2)

        self.assertAlmostEqual(result.step_proxies[0].J_hat_early_t, 0.2)
        self.assertEqual(result.step_proxies[1].J_hat_early_t, 0.0)  # k = 1
        self.assertEqual(
            result.step_proxies[2].J_hat_early_t, 0.0
        )  # outside early block
        for step in result.step_proxies:
            self.assertAlmostEqual(
                step.G_hat_t, step.M_hat_t + step.J_hat_early_t
            )

    def test_tau_is_configurable_and_does_not_change_j(self) -> None:
        result = compute_sample_proxies(_synthetic_trace(), tau=0.5)

        self.assertAlmostEqual(result.M_hat, -math.log(0.5))
        self.assertAlmostEqual(result.J_hat_early, 0.2)
        self.assertAlmostEqual(result.G_hat, -math.log(0.5) + 0.2)
        self.assertEqual(result.tau, 0.5)

    def test_trajectory_blocks_preserve_short_trace_semantics(self) -> None:
        self.assertEqual(trajectory_blocks(0)["early"], (0, 0))
        self.assertEqual(trajectory_blocks(2)["early"], (0, 2))
        self.assertEqual(
            trajectory_blocks(7),
            {
                "full": (0, 7),
                "early": (0, 2),
                "mid": (2, 4),
                "late": (4, 7),
            },
        )

    def test_selection_validation_is_preserved(self) -> None:
        trace = _synthetic_trace()
        trace["step_traces"][0]["selected_positions"][0] = False

        with self.assertRaisesRegex(TraceProxyError, "selected_positions"):
            compute_sample_proxies(trace)

    def test_invalid_tau_is_rejected(self) -> None:
        for tau in (-1.0, float("nan"), float("inf")):
            with self.subTest(tau=tau):
                with self.assertRaisesRegex(ValueError, "tau"):
                    compute_sample_proxies(_synthetic_trace(), tau=tau)


if __name__ == "__main__":
    unittest.main()
