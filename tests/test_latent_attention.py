import unittest
import torch
from schedulers import latent_attention
from schedulers.latent_finite import generate_finite
from test_latent_scheduler import TinyModel


class AttentionChecks(unittest.TestCase):
    def test_uniform_attention_excludes_self_and_known_keys(self):
        q=torch.zeros(1,2,3,4); k=torch.zeros_like(q)
        weights=torch.tensor([[.2,.8,0.]])
        risk=latent_attention.unresolved_read_risk(q,k,weights,query_chunk=1)
        torch.testing.assert_close(risk,torch.tensor([[.8/3,.2/3,1./3]]))
        torch.testing.assert_close(latent_attention.unresolved_read_risk(q,k,weights*0),torch.zeros(1,3))

    def test_padding_bias_and_query_chunk_invariance(self):
        torch.manual_seed(4); q=torch.randn(2,2,5,4); k=torch.randn_like(q)
        weights=torch.rand(2,5); weights[:,-1]=100.
        bias=torch.zeros(2,1,1,5); bias[:,:,:,-1]=float('-inf')
        a=latent_attention.unresolved_read_risk(q,k,weights,bias,1)
        b=latent_attention.unresolved_read_risk(q,k,weights,bias,5)
        torch.testing.assert_close(a,b)
        weights[:,-1]=0
        torch.testing.assert_close(a,latent_attention.unresolved_read_risk(q,k,weights,bias))

    def test_capture_leaves_original_attention_and_restores_method_on_error(self):
        class Block:
            def _scaled_dot_product_attention(self,q,k,v,**kwargs): return v+1
        block=Block(); q=torch.zeros(1,1,2,3)
        with self.assertRaises(RuntimeError):
            with latent_attention.capture_attention_reads([block]) as captured:
                result=block._scaled_dot_product_attention(q,q,q)
                torch.testing.assert_close(result,q+1); self.assertEqual(len(captured),1)
                raise RuntimeError('test cleanup')
        self.assertNotIn('_scaled_dot_product_attention',block.__dict__)
        torch.testing.assert_close(block._scaled_dot_product_attention(q,q,q),q+1)

    def test_zero_weight_recovers_fisher_control_without_attention_capture(self):
        model=TinyModel(); prompt=torch.tensor([[1,2],[3,4]])
        args=dict(steps=3,gen_length=8,mask_id=15,eos_id=14)
        control=generate_finite(model,prompt,torch.ones_like(prompt),gain_beta=0.,**args)
        actual=latent_attention.generate_attention(model,prompt,torch.ones_like(prompt),read_beta=0.,**args)
        self.assertTrue(torch.equal(control,actual))
        self.assertEqual(latent_attention.last_metrics['branch_work_by_row'],[0,0])
