import math
import unittest
import torch

from schedulers.temporal_divergence import temporal_divergence


class PosteriorMovementChecks(unittest.TestCase):
    def test_static_predictions_and_invalid_mask_have_zero_movement(self):
        logits = torch.tensor([[1.,2.,-torch.inf],[3.,4.,-torch.inf]])
        for metric in ('kl','js'):
            torch.testing.assert_close(temporal_divergence(logits,logits,metric),torch.zeros(2),atol=1e-7,rtol=0)

    def test_kl_matches_categorical_definition_and_js_is_bounded(self):
        p, q = torch.tensor([[.99,.01]]),torch.tensor([[.01,.99]])
        kl = temporal_divergence(p.log(),q.log())
        torch.testing.assert_close(kl,(p*(p.log()-q.log())).sum(-1))
        js = temporal_divergence(p.log(),q.log(),'js')
        self.assertGreater(float(js),0.)
        self.assertLessEqual(float(js),math.log(2))
        torch.testing.assert_close(js,temporal_divergence(q.log(),p.log(),'js'))

    def test_gauge_invariance_and_chunk_size(self):
        torch.manual_seed(3)
        p, q = torch.randn(5,20),torch.randn(5,20)
        for metric in ('kl','js'):
            a = temporal_divergence(p,q,metric,2)
            b = temporal_divergence(p+4,q-7,metric,5)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)


if __name__ == '__main__': unittest.main()
