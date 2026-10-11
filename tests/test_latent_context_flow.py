import copy
import unittest

import torch
from torch.nn import functional as F

from schedulers.latent_context_flow import context_flow_capture,context_support
from schedulers.latent_count import forward_fields
from schedulers.latent_allowance import generate_allowance
from test_latent_residual import ActivationModel


class FinalBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_out=torch.nn.Linear(8,8,bias=False)
        self.ff_norm=torch.nn.LayerNorm(8)
        self.ff_proj=torch.nn.Linear(8,12,bias=False)
        self.up_proj=torch.nn.Linear(8,12,bias=False)
        self.ff_out=torch.nn.Linear(12,8,bias=False)

    def _scaled_dot_product_attention(self,q,k,v,attn_mask=None):
        return F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask)

    def forward(self,x):
        q=x[:,None]
        read=self._scaled_dot_product_attention(q,q,q).squeeze(1)
        residual=x+self.attn_out(read)
        z=self.ff_norm(residual)
        return residual+self.ff_out(F.silu(self.ff_proj(z))*self.up_proj(z)),None


def make_model():
    value=ActivationModel()
    value.model.transformer.blocks[-1]=FinalBlock()
    return value


class ContextFlowChecks(unittest.TestCase):
    def test_forward_unchanged_and_partial_gradient_restores_state(self):
        value=make_model();ids=torch.tensor([[2,3,15,15],[4,5,15,15]])
        attention=torch.ones_like(ids)
        flags=[p.requires_grad for p in value.parameters()]
        with torch.no_grad():
            reference=value(ids,attention_mask=attention).logits
            with context_flow_capture(value) as saved:
                out,hidden,*_=forward_fields(value,ids,attention,torch.eye(8),reads=False)
                torch.testing.assert_close(out.logits,reference)
                proposal=out.logits.argmax(-1)
                score,info=context_support(value,saved,proposal,ids[:,2:]==15,ids!=15,2,
                    value.head.weight.detach().mean(0))
                self.assertTrue(torch.isfinite(score).all())
                self.assertGreater(sum(info['gradient_norm']),0)
        self.assertEqual(flags,[p.requires_grad for p in value.parameters()])
        self.assertTrue(all(p.grad is None for p in value.parameters()))
        with torch.no_grad():torch.testing.assert_close(value(ids).logits,reference)

    def test_support_head_common_shift_invariance(self):
        base=make_model();other=copy.deepcopy(base)
        with torch.no_grad():other.head.weight+=torch.randn(1,8)
        ids=torch.tensor([[2,3,15,15]]);results=[]
        for value in (base,other):
            with torch.no_grad(),context_flow_capture(value) as saved:
                out,*_=forward_fields(value,ids,torch.ones_like(ids),torch.eye(8),reads=False)
                score,_=context_support(value,saved,out.logits.argmax(-1),ids[:,2:]==15,ids!=15,2,
                    value.head.weight.detach().mean(0))
                results.append(score)
        torch.testing.assert_close(*results,rtol=1e-5,atol=1e-6)

    def test_partial_gradient_includes_residual_identity(self):
        value=make_model();ids=torch.tensor([[2,3,15,15]])
        transformer=value.model.transformer;block=transformer.blocks[-1]
        with torch.no_grad(),context_flow_capture(value) as saved:
            forward_fields(value,ids,torch.ones_like(ids),torch.eye(8),reads=False)
            direction=torch.randn_like(saved['head'])
            with torch.enable_grad():
                actual=torch.autograd.grad((saved['head']*direction).sum(),saved['ff_input'])[0]
        # Independently rebuild the complete final residual transformation.
        u=saved['ff_input'].detach().requires_grad_(True)
        z=block.ff_norm(u)
        final=transformer.ln_f(u+block.ff_out(F.silu(block.ff_proj(z))*block.up_proj(z)))
        expected=torch.autograd.grad((final*direction).sum(),u)[0]
        torch.testing.assert_close(actual,expected)

    def test_generator_records_reveal_positions_and_partial_backward_cost(self):
        import schedulers.latent_allowance as lab
        value=make_model();calls=[]
        hook=value.register_forward_hook(lambda *_:calls.append(1))
        prompt=torch.tensor([[2,3]])
        result=generate_allowance(value,prompt,torch.ones_like(prompt),steps=3,
            gen_length=4,mask_id=15,ranking='context_flow',controller='singleton',
            metric='uniform',projection=torch.eye(8))
        hook.remove();work=lab.last_metrics
        self.assertEqual(len(calls),3)
        self.assertEqual(work['partial_backward_sequence_evaluations'],3)
        self.assertEqual(sum(len(x) for step in work['reveal_positions'] for x in step),3)
        self.assertEqual(int((result[:,2:]!=15).sum()),3)
        self.assertFalse(work['confidence_scoring'])
        self.assertTrue(torch.equal(result[:,:2],prompt))


if __name__=='__main__':unittest.main()
