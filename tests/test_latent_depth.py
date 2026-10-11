import unittest
from types import SimpleNamespace
import torch
from schedulers import latent_depth
from schedulers.latent_finite import generate_finite
from test_latent_scheduler import TinyModel


class DepthModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module(); self.model.transformer = torch.nn.Module()
        self.model.transformer.ln_f = torch.nn.Identity()
        self.config.block_group_size=1
    def forward(self,ids,attention_mask=None,output_hidden_states=False):
        h = self.embedding(ids); h=h+h.mean(1,keepdim=True); states=[]
        for layer in range(4):
            if output_hidden_states: states.append(h)
            h=h+.05*torch.tanh(h+layer)
        h=self.model.transformer.ln_f(h)
        if output_hidden_states: states.append(h)
        return SimpleNamespace(logits=self.head(h),hidden_states=tuple(states) if output_hidden_states else None)


class DepthChecks(unittest.TestCase):
    def test_zero_depth_weights_recover_control(self):
        model=DepthModel(); prompt=torch.tensor([[1,2],[3,4]])
        args=dict(steps=4,gen_length=8,mask_id=15,eos_id=14)
        control=generate_finite(model,prompt,torch.ones_like(prompt),gain_beta=0.,**args)
        actual=latent_depth.generate_depth(model,prompt,torch.ones_like(prompt),**args)
        self.assertTrue(torch.equal(actual,control))
        self.assertFalse(latent_depth.last_metrics['hidden_states_retained_for_depth'])

    def test_depth_fields_zero_and_common_shift_invariant(self):
        torch.manual_seed(3)
        current=torch.randn(6,4); earlier=torch.randn(6,4); weight=torch.randn(9,4); logits=current@weight.T
        risk,support=latent_depth.depth_fields(logits,current,earlier,weight)
        r2,s2=latent_depth.depth_fields(logits+10,current,earlier,weight+torch.randn(1,4))
        torch.testing.assert_close(risk,r2,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(support,s2,atol=2e-6,rtol=2e-6)
        zero_r,zero_s=latent_depth.depth_fields(logits,current,current,weight)
        self.assertEqual(float(zero_r.abs().sum()+zero_s.abs().sum()),0.)

    def test_actual_proposal_used_when_top_logits_tie(self):
        logits=torch.tensor([[1.,1.,0.]])
        current=torch.tensor([[1.,0.]]); earlier=torch.zeros_like(current)
        weight=torch.tensor([[2.,0.],[-1.,0.],[0.,1.]])
        _,support=latent_depth.depth_fields(logits,current,earlier,weight,3)
        expected=2.-float((logits.softmax(-1)*torch.tensor([[2.,-1.,0.]])).sum())
        self.assertAlmostEqual(float(support),expected,places=6)

    def test_capture_preserves_prompt_and_has_no_additional_model_work(self):
        model=DepthModel(); prompt=torch.tensor([[1,2],[3,4]])
        out=latent_depth.generate_depth(model,prompt,torch.ones_like(prompt),
            steps=3,gen_length=8,mask_id=15,eos_id=14,depth_back=2,
            depth_risk_beta=[0.,.15],depth_support_beta=[.15,0.])
        self.assertTrue(torch.equal(out[:,:2],prompt))
        m=latent_depth.last_metrics
        self.assertEqual(m['forward_work_by_row'],[3,3]); self.assertEqual(m['sequence_evaluations'],6)
        self.assertEqual(m['branch_work_by_row'],[0,0]); self.assertGreater(m['depth_measurements'],0)
        self.assertTrue(all(p.grad is None and p.requires_grad for p in model.parameters()))
