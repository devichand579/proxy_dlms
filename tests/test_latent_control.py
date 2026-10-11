import unittest
from types import SimpleNamespace

import torch

from schedulers.latent_control import latent_prefix, generate_factorized


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.emb = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)
        with torch.no_grad():
            self.head.bias[15] = -100
        self.config = SimpleNamespace(model_type="llada", weight_tying=False)

    def get_output_embeddings(self):
        return self.head

    def get_input_embeddings(self):
        return self.emb

    def forward(self, x, attention_mask=None):
        h = self.emb(x)
        return SimpleNamespace(logits=self.head(h + h.mean(1, keepdim=True)))


class ControllerChecks(unittest.TestCase):
    def test_coupling_reduces_parallel_count(self):
        pos = torch.arange(10)
        rank = -pos.float()
        confidence = torch.full((10,), .995)
        motion = torch.full((10,), .5)
        features = torch.ones(10, 3)
        independent, _, _ = latent_prefix(pos, rank, confidence, motion, features, .05, 0.)
        coupled, _, _ = latent_prefix(pos, rank, confidence, motion, features, .05, 1.)
        self.assertLess(len(coupled), len(independent))

    def test_uncertain_state_forces_singleton_and_records_override(self):
        pos = torch.arange(4)
        selected, forced, cost = latent_prefix(pos, torch.arange(4).float(),
            torch.full((4,), .2), torch.zeros(4), torch.zeros(4, 3), .03, .25)
        self.assertEqual(selected.tolist(), [3])
        self.assertTrue(forced)
        self.assertGreater(cost, .03)

    def test_opposed_responses_are_not_treated_as_independence(self):
        pos = torch.arange(10)
        rank = -pos.float()
        confidence, motion = torch.full((10,), .995), torch.full((10,), .5)
        aligned = torch.ones(10, 3)
        opposed = aligned.clone(); opposed[::2] *= -1
        a, _, _ = latent_prefix(pos, rank, confidence, motion, aligned, .05, 1.)
        b, _, _ = latent_prefix(pos, rank, confidence, motion, opposed, .05, 1.)
        self.assertEqual(len(a), len(b))

    def test_ranking_only_keeps_eb_cardinality_on_identical_history(self):
        model = Tiny()
        prompt = torch.tensor([[1, 2], [1, 2]])
        # One shared history makes the count invariant direct, not trajectory-dependent.
        output = generate_factorized(model, prompt, torch.ones_like(prompt),
            cells=["eb_eb", "eb_latent"], gen_length=8, steps=1, mask_id=15)
        self.assertEqual(int((output[0, 2:] != 15).sum()), int((output[1, 2:] != 15).sum()))
        self.assertTrue(torch.equal(output[:, :2], prompt))

    def test_learned_ranking_keeps_eb_count_and_requires_readout(self):
        class Readout:
            def predict(self, hidden, logits, embeddings, eos_id):
                return torch.linspace(.01, .99, len(hidden))
        model = Tiny()
        prompt = torch.tensor([[1, 2], [1, 2]])
        with self.assertRaisesRegex(ValueError, "require a risk readout"):
            generate_factorized(model, prompt, torch.ones_like(prompt), cells=["eb_eb", "eb_learned"])
        output = generate_factorized(model, prompt, torch.ones_like(prompt),
            cells=["eb_eb", "eb_learned"], gen_length=8, steps=1, mask_id=15,
            risk_readout=Readout())
        self.assertEqual(int((output[0, 2:] != 15).sum()), int((output[1, 2:] != 15).sum()))
        self.assertTrue(torch.equal(output[:, :2], prompt))

    def test_caps_bound_count_and_preserve_prompt(self):
        model = Tiny()
        prompt = torch.tensor([[1, 2], [1, 2]])
        output = generate_factorized(model, prompt, torch.ones_like(prompt),
            cells=["eb_cap1", "eb_cap2"], gen_length=8, steps=2, mask_id=15)
        from schedulers import latent_control
        for counts in latent_control.last_metrics["selected_counts"]:
            self.assertGreaterEqual(counts[0], 1)
            self.assertLessEqual(counts[0], 1)
            self.assertGreaterEqual(counts[1], 1)
            self.assertLessEqual(counts[1], 2)
        self.assertTrue(torch.equal(output[:, :2], prompt))

    def test_zero_motion_penalty_recovers_eb_trajectory(self):
        model = Tiny()
        prompt = torch.tensor([[1, 2], [1, 2]])
        output = generate_factorized(model, prompt, torch.ones_like(prompt),
            cells=["eb_eb", "motion_eb"], gen_length=8, steps=3, mask_id=15,
            motion_beta=0.)
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_content_ranking_changes_positions_without_changing_eb_count(self):
        class Gate:
            def body_probability(self, hidden, logits, embeddings, eos_id):
                return torch.linspace(.01, .99, len(hidden))
        model = Tiny()
        prompt = torch.tensor([[1, 2], [1, 2]])
        output = generate_factorized(model, prompt, torch.ones_like(prompt),
            cells=["eb_eb", "eb_content"], gen_length=8, steps=1, mask_id=15,
            risk_readout=Gate())
        masks = output[:, 2:] != 15
        self.assertEqual(int(masks[0].sum()), int(masks[1].sum()))
        self.assertFalse(torch.equal(masks[0], masks[1]))

    def test_zero_content_and_motion_penalty_recovers_eb(self):
        class Gate:
            def body_probability(self, hidden, logits, embeddings, eos_id):
                return torch.linspace(.01, .99, len(hidden))
        model = Tiny()
        prompt = torch.tensor([[1, 2], [1, 2]])
        output = generate_factorized(model, prompt, torch.ones_like(prompt),
            cells=["eb_eb", "content_entropy"], gen_length=8, steps=3, mask_id=15,
            risk_readout=Gate(), content_beta=0., rank_beta=0.)
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_zero_reveal_value_recovers_eb_without_reference_inputs(self):
        class Value:
            def predict(self, hidden, logits, embeddings, eos_id, masked_hidden):
                return torch.zeros(len(hidden))
        model = Tiny()
        prompt = torch.tensor([[1, 2], [1, 2]])
        with self.assertRaisesRegex(ValueError, "require a reveal-value readout"):
            generate_factorized(model, prompt, torch.ones_like(prompt), cells=["eb_eb", "eb_value"])
        output = generate_factorized(model, prompt, torch.ones_like(prompt),
            cells=["eb_eb", "value_entropy"], gen_length=8, steps=3, mask_id=15,
            value_readout=Value(), rank_beta=0.)
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_exact_posterior_control_recovers_eb_with_zero_penalty(self):
        model = Tiny()
        prompt = torch.tensor([[1,2],[1,2]])
        output = generate_factorized(model,prompt,torch.ones_like(prompt),
            cells=['eb_eb','kl_entropy'],gen_length=8,steps=3,mask_id=15,rank_beta=0.)
        self.assertTrue(torch.equal(output[0],output[1]))

    def test_delayed_count_switch_preserves_fisher_policy_before_switch(self):
        class Readout:
            def predict(self,hidden,logits,embeddings,eos_id):
                return torch.full((len(hidden),),.3)
        model = Tiny()
        prompt = torch.tensor([[1,2],[1,2]])
        output = generate_factorized(model,prompt,torch.ones_like(prompt),
            cells=['fisher_entropy','late_learned_latent'],gen_length=8,steps=3,
            mask_id=15,switch_step=3,risk_readout=Readout())
        self.assertTrue(torch.equal(output[0],output[1]))

    def test_signed_support_distinguishes_confirmation_from_reversal(self):
        from schedulers.latent import _target_temporal_risk
        weight = torch.tensor([[1., 0.], [-1., 0.]])
        current = torch.tensor([[1., 0.]])
        logits = current @ weight.T
        risk_a, support_a = _target_temporal_risk(logits, current, torch.zeros_like(current), weight, 2, True)
        risk_b, support_b = _target_temporal_risk(logits, current, 2 * current, weight, 2, True)
        self.assertTrue(torch.allclose(risk_a, risk_b))
        self.assertGreater(float(support_a[0]), 0)
        self.assertLess(float(support_b[0]), 0)

    def test_common_logit_shift_has_zero_latent_risk_and_support(self):
        from schedulers.latent import _target_temporal_risk, _fisher_project
        weights = torch.ones(4, 2)
        current, previous = torch.tensor([[1000., 1000.]]), torch.zeros(1, 2)
        logits = torch.tensor([[100., 2., 1., 0.]])
        risk, support = _target_temporal_risk(logits, current, previous, weights, 4, True)
        self.assertEqual(float(risk[0]), 0.)
        self.assertEqual(float(support[0]), 0.)
        features = _fisher_project(current, logits.softmax(-1)[0], weights)
        self.assertTrue(torch.equal(features, torch.zeros_like(features)))


if __name__ == "__main__":
    unittest.main()
