import copy
import itertools
import unittest
import torch

from schedulers import latent_flow,latent_residual
from test_latent_residual import ActivationModel


class FlowTests(unittest.TestCase):
    def test_spectral_psd_objective_equals_direct_transport_minus_interference(self):
        contrast = torch.tensor([[0.,2.,.1],[.3,0.,1.],[.1,.2,0.]],dtype=torch.double)
        defect = contrast*.2
        gain,kernel = latent_flow.flow_objective(contrast,defect,torch.arange(3),
            incoming_weight=.5,consistency_weight=.15,progress_reward=.05)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(kernel).min()),-1e-10)
        scale = contrast.sum(1).mean()
        for bits in itertools.product([0.,1.],repeat=3):
            v = torch.tensor(bits,dtype=torch.double)
            direct = ((contrast.sum(1)-.5*contrast.sum(0)-.15*defect.sum(1))/scale+.05)@v
            direct -= v@contrast@v/scale
            spectral = gain@v-.5*v@kernel@v
            self.assertAlmostEqual(float(direct),float(spectral),places=9)

    def test_independent_latent_dependency_components_commit_upstream_nodes_together(self):
        contrast = torch.tensor([[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.]])
        gain,kernel = latent_flow.flow_objective(contrast,torch.zeros_like(contrast),torch.arange(4))
        chosen,_ = latent_residual.subset_optimum(gain,kernel)
        self.assertEqual(chosen.tolist(),[0,2])

    def test_head_temperature_does_not_change_latent_policy_or_count(self):
        first = ActivationModel();second = copy.deepcopy(first);second.logit_scale = .07
        prompt = torch.tensor([[1,2],[3,4]])
        args = dict(steps=4,gen_length=5,mask_id=15,eos_id=14,candidates=3,branch_batch_size=4)
        original = latent_flow.generate_flow(first,prompt,torch.ones_like(prompt),**args)
        counts = latent_flow.last_metrics['reveal_counts']
        actual = latent_flow.generate_flow(second,prompt,torch.ones_like(prompt),**args)
        self.assertTrue(torch.equal(original,actual))
        self.assertTrue(torch.equal(actual[:,:2],prompt))
        self.assertEqual(counts,latent_flow.last_metrics['reveal_counts'])
        m = latent_flow.last_metrics
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])
        self.assertEqual(m['branch_work_by_row'],m['probe_control_work_by_row'])
        self.assertEqual(m['source_value_weighting'],'uniform')
        self.assertFalse(m['confidence_scoring'])
        self.assertFalse(m['entropy_scoring'])
        self.assertFalse(m['eb_count'])

    def test_activation_metric_preserves_distances_under_rotation_and_rejects_nuisance_axes(self):
        states = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,2.,0.],[0.,-2.,0.]])
        basis = latent_flow.activation_metric(states)
        nuisance = torch.tensor([[0.,0.,7.]])
        self.assertLess(float((nuisance@basis).abs().max()),1e-7)
        rotation,_ = torch.linalg.qr(torch.tensor([[1.,2.,3.],[2.,-2.,1.],[3.,1.,-1.]]))
        rotated = latent_flow.activation_metric(states@rotation)
        delta = torch.tensor([[.4,.7,.2]])
        torch.testing.assert_close((delta@basis).square().sum(),(delta@rotation@rotated).square().sum(),rtol=1e-5,atol=1e-7)

    def test_checkpoint_resume_matches_uninterrupted_latent_policy_and_work(self):
        model = ActivationModel();prompt = torch.tensor([[1,2],[3,4]])
        args = dict(steps=4,gen_length=6,mask_id=15,eos_id=14,candidates=3,
                    branch_batch_size=4,geometry='pca')
        checkpoints = []
        def remember(state):checkpoints.append(copy.deepcopy(state))
        expected = latent_flow.generate_flow(model,prompt,torch.ones_like(prompt),step_checkpoint=remember,**args)
        expected_metrics = copy.deepcopy(latent_flow.last_metrics)
        self.assertGreater(len(checkpoints),1)
        actual = latent_flow.generate_flow(model,prompt,torch.ones_like(prompt),resume_state=checkpoints[0],**args)
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(expected_metrics,latent_flow.last_metrics)

    def test_unused_controls_can_be_removed_without_changing_policy(self):
        model = ActivationModel();prompt = torch.tensor([[1,2],[3,4]])
        args = dict(steps=4,gen_length=6,mask_id=15,eos_id=14,candidates=3,
                    branch_batch_size=4,geometry='pca',consistency_weight=0.)
        first = latent_flow.generate_flow(model,prompt,torch.ones_like(prompt),**args)
        original = copy.deepcopy(latent_flow.last_metrics)
        second = latent_flow.generate_flow(model,prompt,torch.ones_like(prompt),evaluate_controls=False,**args)
        actual = latent_flow.last_metrics
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(original['reveal_counts'],actual['reveal_counts'])
        self.assertEqual(actual['probe_control_work_by_row'],[0,0])
        self.assertEqual(original['sequence_evaluations']-sum(original['probe_control_work_by_row']),actual['sequence_evaluations'])
        with self.assertRaises(ValueError):
            latent_flow.generate_flow(model,prompt,torch.ones_like(prompt),consistency_weight=.15,evaluate_controls=False)


if __name__=='__main__':unittest.main()
