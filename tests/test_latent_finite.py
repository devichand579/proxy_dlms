import unittest
import torch

from schedulers import latent_finite
from schedulers.latent_control import generate_factorized
from test_latent_scheduler import TinyModel


class FiniteRevealChecks(unittest.TestCase):
    def test_zero_gain_recovers_fisher_entropy_without_extra_work(self):
        model = TinyModel(); prompt = torch.tensor([[1,2],[3,4]])
        control = generate_factorized(model,prompt,torch.ones_like(prompt),cells=['fisher_entropy']*2,
            steps=4,gen_length=8,mask_id=15,eos_id=14)
        actual = latent_finite.generate_finite(model,prompt,torch.ones_like(prompt),
            steps=4,gen_length=8,mask_id=15,eos_id=14,gain_beta=0.)
        self.assertTrue(torch.equal(control,actual))
        self.assertEqual(latent_finite.last_metrics['branch_work_by_row'],[0,0])

    def test_branches_preserve_prompt_and_are_counted_per_sample(self):
        model = TinyModel(); prompt = torch.tensor([[1,2],[3,4]])
        actual = latent_finite.generate_finite(model,prompt,torch.ones_like(prompt),
            steps=3,gen_length=8,mask_id=15,eos_id=14,gain_beta=[0.,.5],
            candidates=2,probe_every=1,branch_batch_size=1)
        self.assertTrue(torch.equal(actual[:,:2],prompt))
        m = latent_finite.last_metrics
        self.assertEqual(m['branch_work_by_row'][0],0)
        self.assertGreater(m['branch_work_by_row'][1],0)
        self.assertEqual(m['forward_work_by_row'][0],m['steps_taken'])
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])
        self.assertFalse(m['source_entropy_counted'])

    def test_density_weights_are_computed_and_handle_identical_states(self):
        hidden = torch.ones(4,3); logits = torch.randn(4,6); weight = torch.randn(6,3)
        actual = latent_finite.latent_density_weights(hidden,logits,weight)
        torch.testing.assert_close(actual,torch.ones(4))
        varied = torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,2.,1.],[0.,2.,1.]])
        actual = latent_finite.latent_density_weights(varied,logits,weight)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual>0).all())
        torch.testing.assert_close(actual.mean(),torch.tensor(1.))

    def test_matched_controls_preserve_decisions_and_count_extra_work(self):
        model = TinyModel(); prompt = torch.tensor([[1,2],[3,4]])
        params = dict(steps=3,gen_length=8,mask_id=15,eos_id=14,
                      gain_beta=[0.,.5],candidates=2,probe_every=1,branch_batch_size=2)
        original = latent_finite.generate_finite(model,prompt,torch.ones_like(prompt),**params)
        old = dict(latent_finite.last_metrics)
        matched = latent_finite.generate_finite(model,prompt,torch.ones_like(prompt),
            matched_probe_control=True,**params)
        self.assertTrue(torch.equal(original,matched))
        m = latent_finite.last_metrics
        self.assertEqual(m['probe_control_work_by_row'],m['branch_work_by_row'])
        self.assertEqual(m['sequence_evaluations'],old['sequence_evaluations']+sum(m['probe_control_work_by_row']))
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])


if __name__=='__main__': unittest.main()
