import unittest
import torch
from schedulers import latent_conditioning
from schedulers.latent_finite import generate_finite
from test_latent_scheduler import TinyModel


class ConditioningChecks(unittest.TestCase):
    def test_zero_weight_is_fisher_control_without_extra_evaluations(self):
        model = TinyModel(); prompt = torch.tensor([[1,2],[3,4]])
        common = dict(steps=4,gen_length=8,mask_id=15,eos_id=14)
        control = generate_finite(model,prompt,torch.ones_like(prompt),gain_beta=0.,**common)
        actual = latent_conditioning.generate_conditioning(model,prompt,torch.ones_like(prompt),context_beta=0.,**common)
        self.assertTrue(torch.equal(actual,control))
        self.assertEqual(latent_conditioning.last_metrics['branch_work_by_row'],[0,0])

    def test_independent_targets_have_zero_conditioning_evidence(self):
        model = TinyModel(coupled=False); prompt = torch.tensor([[1,2],[3,4]])
        out = latent_conditioning.generate_conditioning(model,prompt,torch.ones_like(prompt),
            steps=3,gen_length=8,mask_id=15,eos_id=14,context_beta=[0.,.5],branch_batch_size=2)
        m = latent_conditioning.last_metrics
        self.assertTrue(torch.equal(out[:,:2],prompt))
        self.assertLess(m['evidence_abs_sum'],1e-6)
        self.assertEqual(m['branch_work_by_row'],[0,3])
        self.assertEqual(m['probe_control_work_by_row'],[0,3])
        self.assertEqual(m['forward_work_by_row'],[3,9])
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_fixed_infilling_context_is_preserved_and_only_initial_context_removed(self):
        class RecordingModel(TinyModel):
            def __init__(self): super().__init__(); self.inputs=[]
            def forward(self,ids,**kwargs):
                self.inputs.append(ids.clone()); return super().forward(ids,**kwargs)
        model = RecordingModel(); canvas = torch.tensor([[1,15,2,15,3,15]])
        prompt = canvas[:,:0]
        out = latent_conditioning.generate_conditioning(model,prompt,prompt,
            initial_canvas=canvas,full_attention=torch.ones_like(canvas),steps=3,gen_length=6,
            mask_id=15,eos_id=14,context_beta=.5,branch_batch_size=2)
        known = canvas!=15
        self.assertTrue(torch.equal(out[known],canvas[known]))
        for control,paired in zip(model.inputs[::2],model.inputs[1::2]):
            self.assertTrue(torch.equal(paired[:1],control))
            self.assertTrue(bool((paired[1:][known]==15).all()))
            self.assertTrue(torch.equal(paired[1:][~known],control[~known]))

    def test_no_context_skips_useless_probes(self):
        model = TinyModel(); prompt = torch.empty((1,0),dtype=torch.long)
        latent_conditioning.generate_conditioning(model,prompt,prompt,
            steps=3,gen_length=8,mask_id=15,eos_id=14,context_beta=.5)
        self.assertEqual(latent_conditioning.last_metrics['branch_work_by_row'],[0])
