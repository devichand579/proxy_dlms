import copy
import unittest
import torch
from torch.nn import functional as F

from schedulers import latent_count
from test_latent_residual import ActivationModel,ToyBlock


class ReadBlock(ToyBlock):
    def _scaled_dot_product_attention(self,q,k,v,attn_mask=None):
        return F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask)

    def forward(self,x):
        q=x[:,None]
        read=self._scaled_dot_product_attention(q,q,q).squeeze(1)
        return super().forward(x+.05*read)


def model():
    value=ActivationModel()
    value.model.transformer.blocks=torch.nn.ModuleList([ReadBlock() for _ in range(5)])
    return value


class CountChecks(unittest.TestCase):
    def test_guard_handles_independence_copy_and_higher_order_parity(self):
        sizes=[1,2,4,7,8]
        independent,_=latent_count.batch_count(sizes,[0.]*5,8,128)
        copied,_=latent_count.batch_count(sizes,[0.,1.,1.,1.,1.],8,128)
        parity,_=latent_count.batch_count(sizes,[0.,0.,0.,0.,1.],8,128)
        self.assertEqual((independent,copied,parity),(8,1,7))

    def test_conservative_envelope_and_recorded_pacing_override(self):
        sizes=[1,2,4,8]
        free,diag=latent_count.batch_count(sizes,[0.,.2,.01,.001],16,4,budget=.1)
        self.assertEqual(free,1)
        forced,diag=latent_count.batch_count(sizes,[0.,.2,.01,.001],16,4,budget=.1,paced=True)
        self.assertEqual(forced,4);self.assertTrue(diag['forced_pacing'])
        self.assertGreater(diag['error_envelope'][2],diag['threshold'])

    def test_geometry_ignores_common_decoder_shift_and_temperature_scaling(self):
        torch.manual_seed(3);weight=torch.randn(23,7)
        original=latent_count.decoder_geometry(weight,width=8,chunk=6)
        shifted=latent_count.decoder_geometry(weight+torch.randn(1,7),width=8,chunk=6)
        scaled=latent_count.decoder_geometry(weight*.07,width=8,chunk=6)
        torch.testing.assert_close(original,shifted,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(original,scaled,rtol=1e-5,atol=1e-6)

    def test_known_context_attention_masks_padding(self):
        q=torch.zeros(1,1,3,4);known=torch.tensor([[1.,0.,0.]])
        actual=latent_count.known_reads(q,q,known)
        torch.testing.assert_close(actual,torch.full((1,3),1/3))
        bias=torch.tensor([[[[0.,0.,float('-inf')]]]])
        torch.testing.assert_close(latent_count.known_reads(q,q,known,bias),torch.full((1,3),.5))

    def test_diagnostic_fisher_adapter_accepts_single_sequence_fields(self):
        from schedulerlens.latent_count_bench import diagnostic_fisher
        logits=torch.randn(4,16);hidden=torch.randn(4,8);weight=torch.randn(16,8)
        initial=diagnostic_fisher(logits,hidden,None,weight)
        subsequent=diagnostic_fisher(logits,hidden,hidden*.95,weight)
        self.assertEqual(initial.shape,(4,));self.assertTrue(torch.isfinite(subsequent).all())

    def test_confidence_rescaling_preserves_scorer_count_and_work(self):
        first=model();second=copy.deepcopy(first);second.logit_scale=.07
        prompt=torch.tensor([[1,2],[3,4]])
        args=dict(steps=4,gen_length=6,mask_id=15,eos_id=14,max_count=4,branch_batch_size=3)
        a=latent_count.generate_count(first,prompt,torch.ones_like(prompt),**args)
        first_metrics=copy.deepcopy(latent_count.last_metrics)
        b=latent_count.generate_count(second,prompt,torch.ones_like(prompt),**args)
        last=latent_count.last_metrics
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(b[:,:2],prompt))
        self.assertEqual(first_metrics['reveal_counts'],last['reveal_counts'])
        self.assertEqual(sum(last['forward_work_by_row']),last['sequence_evaluations'])
        self.assertEqual(last['branch_work_by_row'],last['probe_control_work_by_row'])
        self.assertFalse(last['confidence_scoring']);self.assertFalse(last['entropy_scoring']);self.assertFalse(last['eb_count'])
        self.assertTrue(all(p.grad is None for p in first.parameters()))
        self.assertTrue(all('_scaled_dot_product_attention' not in block.__dict__ for block in first.model.transformer.blocks))

    def test_candidate_gradient_metric_is_confidence_invariant(self):
        first=model();second=copy.deepcopy(first);second.logit_scale=.07
        prompt=torch.tensor([[1,2],[3,4]])
        args=dict(steps=3,gen_length=6,mask_id=15,eos_id=14,max_count=4,
                  branch_batch_size=3,growth_metric='candidate_gradients')
        a=latent_count.generate_count(first,prompt,torch.ones_like(prompt),**args)
        counts=copy.deepcopy(latent_count.last_metrics['reveal_counts'])
        b=latent_count.generate_count(second,prompt,torch.ones_like(prompt),**args)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(counts,latent_count.last_metrics['reveal_counts'])

    def test_resume_preserves_attention_rank_and_conditional_growth_count(self):
        value=model();prompt=torch.tensor([[1,2],[3,4]])
        args=dict(steps=4,gen_length=7,mask_id=15,eos_id=14,max_count=4,
                  branch_batch_size=3,growth_metric='candidate_gradients')
        states=[]
        a=latent_count.generate_count(value,prompt,torch.ones_like(prompt),step_checkpoint=lambda s:states.append(copy.deepcopy(s)),**args)
        counts=copy.deepcopy(latent_count.last_metrics['reveal_counts'])
        b=latent_count.generate_count(value,prompt,torch.ones_like(prompt),resume_state=states[0],**args)
        self.assertTrue(torch.equal(a,b));self.assertEqual(counts,latent_count.last_metrics['reveal_counts'])


if __name__=='__main__':unittest.main()
