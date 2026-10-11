import unittest
from types import SimpleNamespace

import torch

from schedulers import latent


class TinyModel(torch.nn.Module):
    def __init__(self, coupled=True):
        super().__init__()
        torch.manual_seed(10)
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)
        with torch.no_grad():
            self.head.bias[14:] = -100
        self.config = SimpleNamespace(model_type="llada", weight_tying=False)
        self.coupled = coupled

    def get_output_embeddings(self):
        return self.head

    @property
    def device(self):
        return self.embedding.weight.device

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, ids, attention_mask=None, inputs_embeds=None):
        hidden = self.embedding(ids) if inputs_embeds is None else inputs_embeds
        if self.coupled:
            hidden = hidden + hidden.mean(1, keepdim=True)
        return SimpleNamespace(logits=self.head(hidden))


class LatentSchedulerTests(unittest.TestCase):
    def test_all_variants_keep_prompt_and_complete_fixed_budget(self):
        for variant in (v for v in latent.VARIANTS if not v.startswith("eb_")):
            with self.subTest(variant=variant):
                model = TinyModel()
                prompt = torch.tensor([[1, 2], [3, 4]])
                out, steps, blocks = latent.generate_latent(model, prompt, torch.ones_like(prompt),
                    variant=variant, gen_length=12, block_length=12, steps=6,
                    probe_every=1, mask_id=15, eos_id=14, eot_id=None, use_eos_harness=False)
                self.assertTrue(torch.equal(out[:, :2], prompt))
                self.assertEqual(int((out[:, 2:] == 15).sum()), 0)
                self.assertEqual((steps, blocks), (6, 1))
                metrics = latent.last_metrics
                expected = 2 * steps + metrics["response_sequence_evaluations"] if variant.startswith("response_") else 2 * (metrics["forward_calls"] + metrics["probe_steps"])
                self.assertEqual(metrics["sequence_evaluations"], expected)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_shared_target_gradient_is_zero_for_independent_positions(self):
        model = TinyModel(coupled=False)
        ids = torch.full((1, 6), 15)
        metrics = {"forward_calls": 0, "sequence_evaluations": 0, "backward_passes": 0, "backward_sequence_evaluations": 0}
        gain = latent._gradient_reveal_gain(model, ids, torch.ones_like(ids), ids == 15,
            [torch.tensor([0, 1])], torch.zeros_like(ids), 15, metrics)
        self.assertTrue(torch.equal(gain, torch.zeros_like(gain)))
        self.assertEqual(metrics["backward_passes"], 1)

    def test_zero_gradient_bonus_with_temporal_risk_matches_fisher_control(self):
        model = TinyModel()
        prompt = torch.tensor([[1,2],[3,4]])
        common = dict(gen_length=12,block_length=12,steps=12,mask_id=15,eos_id=14,
                      eot_id=None,use_eos_harness=False,probe_every=1)
        control,_,_ = latent.generate_latent(model,prompt,torch.ones_like(prompt),variant='eb_target_temporal',**common)
        combined,_,_ = latent.generate_latent(model,prompt,torch.ones_like(prompt),variant='eb_gradient_gain',
            gain_beta=0.,gradient_temporal=True,gradient_pool_min=8,**common)
        self.assertTrue(torch.equal(control,combined))
        self.assertEqual(latent.last_metrics['backward_passes'],0)

    def test_probe_microbatch_accounts_for_active_rows_and_preserves_grad_flags(self):
        model = TinyModel()
        prompt = torch.tensor([[1,2],[3,4]])
        latent.generate_latent(model,prompt,torch.ones_like(prompt),variant='eb_gradient_gain',
            gen_length=12,block_length=12,steps=12,mask_id=15,eos_id=14,eot_id=None,
            use_eos_harness=False,probe_every=1,gain_beta=[0.,.25],gradient_probe_batch_size=1,
            gradient_pool_min=[8,8],gradient_temporal=[True,True])
        m = latent.last_metrics
        self.assertEqual(m['backward_work_by_row'][0],0)
        self.assertGreater(m['backward_work_by_row'][1],0)
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])
        self.assertEqual(sum(m['backward_work_by_row']),m['backward_sequence_evaluations'])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_damage_penalty_is_monotone_and_readiness_does_not_hide_negative_gain(self):
        gain = torch.tensor([-2.,2.,2.])
        confidence = torch.tensor([.5,.5,.99])
        risk = torch.full((3,),2.)
        base = latent._gradient_score_bonus(gain,confidence,risk)
        penalized = latent._gradient_score_bonus(gain,confidence,risk,risk_eta=1.)
        gated = latent._gradient_score_bonus(gain,confidence,risk,risk_eta=1.,readiness_gate=1.)
        self.assertTrue(torch.all(penalized<=base))
        self.assertTrue(torch.all(gated<=penalized))
        self.assertEqual(float(gated[0]),float(penalized[0]))
        self.assertGreater(float(penalized[2]),float(penalized[1]))

    def test_eb_latent_variants_progress_and_complete_with_sufficient_steps(self):
        for variant in (v for v in latent.VARIANTS if v.startswith("eb_")):
            model = TinyModel()
            prompt = torch.tensor([[1, 2], [3, 4]])
            out, steps, _ = latent.generate_latent(model, prompt, torch.ones_like(prompt),
                variant=variant, gen_length=12, block_length=12, steps=12,
                probe_every=1, mask_id=15, eos_id=14, eot_id=None, use_eos_harness=False)
            self.assertEqual(int((out[:, 2:] == 15).sum()), 0)
            self.assertLessEqual(steps, 12)
            self.assertEqual(latent.last_metrics["cardinality"], "eb_entropy_budget")

    def test_eb_prefix_matches_existing_python_rule_with_stable_ties(self):
        positions = torch.arange(7)
        ranking = torch.tensor([.7, .7, .6, .5, .4, .2, .1])
        entropy = torch.tensor([.04, .03, .06, .01, 1., .04, .1])
        selected, history = [], []
        for pos in sorted(positions.tolist(), key=lambda p: float(ranking[p]), reverse=True):
            history.append(float(entropy[pos]))
            if sum(history) - max(history) <= .1:
                selected.append(pos)
            else:
                break
        self.assertEqual(latent._eb_prefix(positions, ranking, entropy).tolist(), selected)

    def test_existing_eb_sampler_accepts_infilling_without_changing_visible_tokens(self):
        from schedulers.eb_sampler import generate_EB
        model = TinyModel()
        canvas = torch.tensor([[1, 15, 2, 15, 15, 14]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 0]])
        out, _, _ = generate_EB(model, canvas[:, :1], attention[:, :1],
            initial_canvas=canvas, canvas_attention_mask=attention,
            gen_length=5, block_length=5, steps=3, mask_id=15, eos_id=14,
            eot_id=None, use_eos_harness=False, upload_to_wandb=False)
        self.assertTrue(torch.equal(out[:, [0, 2, 5]], canvas[:, [0, 2, 5]]))
        self.assertEqual(int((out == 15).sum()), 0)

    def test_vectorized_eb_control_matches_existing_sampler(self):
        from schedulers.eb_sampler import generate_EB
        model = TinyModel()
        prompt = torch.tensor([[1, 2], [3, 4]])
        common = dict(gen_length=12, block_length=12, steps=12,
                      mask_id=15, eos_id=14, eot_id=None, use_eos_harness=False)
        original, original_steps, _ = generate_EB(model, prompt, torch.ones_like(prompt), upload_to_wandb=False, **common)
        control, control_steps, _ = latent.generate_latent(model, prompt, torch.ones_like(prompt), variant="eb_confidence", **common)
        self.assertTrue(torch.equal(original, control))
        self.assertEqual(original_steps, control_steps)

    def test_infilling_preserves_visible_scaffold_and_padding(self):
        model = TinyModel()
        canvas = torch.tensor([[1, 15, 2, 15, 15, 14]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 0]])
        out, steps, _ = latent.generate_latent(model, canvas[:, :1], attention[:, :1],
            initial_canvas=canvas, canvas_attention_mask=attention,
            variant="fisher_diversity", gen_length=5, block_length=5, steps=3,
            mask_id=15, eos_id=14, eot_id=None, use_eos_harness=False)
        self.assertTrue(torch.equal(out[:, [0, 2, 5]], canvas[:, [0, 2, 5]]))
        self.assertEqual(int((out == 15).sum()), 0)
        self.assertEqual(steps, 3)

    def test_fisher_projection_preserves_weighted_logit_variance(self):
        torch.manual_seed(5)
        hidden, weights = torch.randn(3, 8), torch.randn(7, 8)
        p = torch.rand(7); p /= p.sum()
        projected = latent._fisher_project(hidden, p, weights)
        changes = hidden @ weights.T
        expected = (p * changes.square()).sum(-1) - (changes @ p).square()
        self.assertTrue(torch.allclose(projected.square().sum(-1), expected, atol=1e-5))

    def test_diversity_selects_distinct_positions_at_fixed_cardinality(self):
        features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., -1.]])
        chosen = latent._greedy(torch.tensor([0., -.01, -.02, -.03]), features, 2, diversity_beta=.25)
        self.assertEqual(chosen.tolist(), [0, 2])

    def test_probe_utility_excludes_all_proposed_sources(self):
        targets = torch.tensor([False, False, True])
        a = latent._probe_utility(torch.tensor([5., 3., 2.]), torch.tensor([0., 0., 1.]), targets, torch.tensor([-.2]))
        b = latent._probe_utility(torch.tensor([50., 30., 2.]), torch.tensor([12., 12., 1.]), targets, torch.tensor([-.2]))
        self.assertEqual(float(a), float(b))


if __name__ == "__main__":
    unittest.main()
