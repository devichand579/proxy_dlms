import unittest
from types import SimpleNamespace
import torch
from schedulers import latent_consistency
from schedulers.latent_finite import generate_finite
from test_latent_scheduler import TinyModel


class ConsistencyChecks(unittest.TestCase):
    def test_conflicting_commitments_are_serialized_and_values_not_replaced(self):
        class ConflictingModel(TinyModel):
            def forward(self,ids,attention_mask=None):
                base=super().forward(ids,attention_mask=attention_mask)
                logits=torch.full_like(base.logits,-10.)
                choice=torch.where((ids[:,2:]==1).any(-1),2,1)
                logits.scatter_(-1,choice[:,None,None].expand(-1,ids.shape[1],1),10.)
                return SimpleNamespace(logits=logits)
        model=ConflictingModel(); prompt=torch.tensor([[3,4]])
        result=latent_consistency.generate_consistency(model,prompt,torch.ones_like(prompt),
            steps=2,gen_length=2,mask_id=15,eos_id=14,branch_batch_size=4,stability_beta=0.)
        self.assertEqual(result[0,2:].tolist(),[1,2])
        m=latent_consistency.last_metrics
        self.assertEqual(m['conflicts_by_row'],[2]); self.assertEqual(m['count_reductions_by_row'],[1])
        self.assertEqual(m['singleton_fallbacks_by_row'],[1]); self.assertEqual(m['reveal_counts'],[[1],[1]])

    def test_batch_shape_mismatches_are_not_attributed_to_reveals(self):
        class ShapeSensitiveModel(TinyModel):
            def forward(self,ids,attention_mask=None):
                base=super().forward(ids,attention_mask=attention_mask)
                logits=torch.full_like(base.logits,-10.); logits[...,1 if len(ids)==1 else 2]=10.
                return SimpleNamespace(logits=logits)
        model=ShapeSensitiveModel(); prompt=torch.tensor([[3,4]])
        result=latent_consistency.generate_consistency(model,prompt,torch.ones_like(prompt),
            steps=1,gen_length=2,mask_id=15,eos_id=14,branch_batch_size=4,stability_beta=0.)
        self.assertEqual(result[0,2:].tolist(),[1,1])
        m=latent_consistency.last_metrics
        self.assertEqual(m['conflicts_by_row'],[0]); self.assertEqual(m['count_reductions_by_row'],[0])
        self.assertEqual(m['numerical_mismatches_by_row'],[2])

    def test_filter_preserves_stable_order_and_singleton_fallback(self):
        pos=torch.tensor([4,2,8,3])
        chosen=latent_consistency.stable_subset(pos,torch.tensor([False,True,False,True]))
        self.assertEqual(chosen.tolist(),[4,8])
        all_bad=torch.ones(4,dtype=torch.bool)
        self.assertEqual(latent_consistency.stable_subset(pos,all_bad).tolist(),[4])
        self.assertEqual(latent_consistency.stable_subset(pos,torch.tensor([False,True,False,False]),'singleton').tolist(),[4])

    def test_disabled_is_exact_control_without_extra_work(self):
        model=TinyModel(); prompt=torch.tensor([[1,2],[3,4]])
        params=dict(steps=3,gen_length=8,mask_id=15,eos_id=14)
        expected=generate_finite(model,prompt,torch.ones_like(prompt),gain_beta=0.,**params)
        actual=latent_consistency.generate_consistency(model,prompt,torch.ones_like(prompt),enabled=False,**params)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(latent_consistency.last_metrics['branch_work_by_row'],[0,0])

    def test_independent_batch_is_not_restricted_and_compute_is_counted(self):
        model=TinyModel(coupled=False)
        with torch.no_grad():
            model.head.weight.zero_(); model.head.bias.fill_(-10.); model.head.bias[1]=0.
        prompt=torch.tensor([[1,2],[3,4]])
        actual=latent_consistency.generate_consistency(model,prompt,torch.ones_like(prompt),
            steps=2,gen_length=8,mask_id=15,eos_id=14,enabled=[False,True],branch_batch_size=4)
        m=latent_consistency.last_metrics
        self.assertTrue(torch.equal(actual[:,:2],prompt))
        self.assertEqual(int((actual[:,2:]==15).sum()),0)
        self.assertEqual(m['conflicts_by_row'],[0,0])
        self.assertEqual(m['branch_work_by_row'],[0,2])
        self.assertEqual(m['probe_control_work_by_row'],[0,2])
        self.assertEqual(m['forward_work_by_row'],[1,5])
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
