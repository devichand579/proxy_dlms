import tempfile
import unittest
from pathlib import Path

import torch

from schedulers.latent_risk import RiskHead, RiskReadout, risk_features


class RiskReadoutChecks(unittest.TestCase):
    def test_features_ignore_common_logit_shift(self):
        torch.manual_seed(5)
        hidden, embeddings = torch.randn(3, 4), torch.randn(6, 4)
        logits = torch.randn(3, 6)
        hp, ep = torch.randn(4, 2), torch.randn(4, 2)
        before = risk_features(hidden, logits, embeddings, hp, ep, 5)
        after = risk_features(hidden, logits + 10, embeddings, hp, ep, 5)
        torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)

    def test_body_gate_preserves_body_confidence_and_penalizes_bad_tail(self):
        # Position 0 proposes harmless whitespace; position 1 proposes code.
        logits = torch.tensor([[2., 0., -1.], [0., 2., -1.]])
        embeddings, hidden = torch.zeros(3, 2), torch.zeros(2, 2)
        confidence_error = 1 - logits.softmax(-1).amax(-1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            for body_probability, expected in (
                (1., confidence_error), (0., torch.tensor([0., 1.]))
            ):
                head = RiskHead(4)
                for parameter in head.parameters():
                    parameter.data.zero_()
                head.network[-1].bias.data.fill_(100 if body_probability else -100)
                torch.save({"head": head.state_dict(), "mean": torch.zeros(4),
                    "scale": torch.ones(4), "h_projection": torch.empty(2, 0),
                    "e_projection": torch.empty(2, 0), "harmless": torch.tensor([True, False, True]),
                    "metadata": {"feature_mode": "body_gate"}}, path)
                readout = RiskReadout(path, device="cpu")
                torch.testing.assert_close(readout.predict(hidden, logits, embeddings, 2), expected)


if __name__ == "__main__":
    unittest.main()
