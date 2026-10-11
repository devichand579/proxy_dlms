import unittest
import math
import torch
from schedulers import latent_information
from schedulers.latent_finite import generate_finite
from test_latent_scheduler import TinyModel


class InformationChecks(unittest.TestCase):
    def test_depth_combination_uses_existing_forward_and_preserves_prompt(self):
        from test_latent_depth import DepthModel
        model=DepthModel(); prompt=torch.tensor([[1,2],[3,4]])
        result=latent_information.generate_information(model,prompt,torch.ones_like(prompt),
            steps=3,gen_length=8,mask_id=15,eos_id=14,candidates=2,probe_every=1,
            branch_batch_size=4,depth_risk_beta=[0.,.15],incoming_beta=.15)
        m=latent_information.last_metrics
        self.assertTrue(torch.equal(result[:,:2],prompt))
        self.assertEqual(m['branch_work_by_row'],m['probe_control_work_by_row'])
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])
        for row in range(2):
            self.assertEqual(m['forward_work_by_row'][row],m['steps_taken']+2*m['branch_work_by_row'][row])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
    def test_measured_source_support_is_a_full_joint_information_contribution(self):
        conditional = torch.tensor([[[[.8,.1,.1]],[[.1,.7,.2]],[[.3,.3,.4]]]],dtype=torch.float64)
        rho = torch.tensor([[.14,.06,.8]],dtype=torch.float64)
        baseline = torch.tensor([[[.3,.3,.4]]],dtype=torch.float64)
        full,_ = latent_information.response_information(conditional,rho,baseline)
        supported,_ = latent_information.response_information(conditional[:,:2],rho[:,:2]/.2,baseline)
        self.assertTrue(bool((full+1e-12>=.2*supported).all()))

    def test_count_surprise_relaxes_independent_positions_and_conflict_restricts(self):
        pos = torch.arange(4); score = torch.tensor([4.,3.,2.,1.])
        entropy = torch.full((4,),.2); confidence = torch.full((4,),.99)
        zero = torch.zeros(4,4)
        full = latent_information.probe_conflict_prefix(pos,score,entropy,confidence,pos,zero)
        self.assertEqual(len(full),4)
        conflict = torch.full((4,4),.2); conflict.fill_diagonal_(0)
        cautious = latent_information.probe_conflict_prefix(pos,score,entropy,confidence,pos,conflict)
        self.assertEqual(len(cautious),1)
        empty = pos[:0]
        fallback = latent_information.probe_conflict_prefix(pos,score,entropy,confidence,empty,torch.zeros(0,0))
        from schedulers.latent import _eb_prefix
        self.assertTrue(torch.equal(fallback,_eb_prefix(pos,score,entropy)))

    def test_count_only_probes_even_with_zero_rank_bonus_and_changes_count(self):
        model = TinyModel(coupled=False)
        with torch.no_grad():
            model.head.weight.zero_(); model.head.bias.fill_(math.log(.02/(14*.98))); model.head.bias[1]=0.
        prompt = torch.tensor([[1,2]])
        actual = latent_information.generate_information(model,prompt,torch.ones_like(prompt),
            steps=2,gen_length=8,mask_id=15,eos_id=14,gain_beta=0.,incoming_beta=0.,
            count_mode='probe_surprise',candidates=8,probe_every=1,branch_batch_size=4,target_vocab=4)
        self.assertEqual(int((actual[:,2:]==15).sum()),0)
        m = latent_information.last_metrics
        self.assertGreater(m['count_changes'],0)
        self.assertGreater(m['count_extra_reveals'],0)
        self.assertGreater(m['branch_work_by_row'][0],0)
        self.assertEqual(m['count_changes'],sum(m['count_changes_by_row']))
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])
        self.assertTrue(torch.equal(actual[:,:2],prompt))

    def test_prefix_weights_use_committed_marker_and_preserve_prompt_boundary(self):
        canvas = torch.tensor([14,1,2,15,4,14,15,7])
        pos = torch.tensor([3,6])
        actual = latent_information.decoded_prefix_weights(canvas,pos,3,14)
        torch.testing.assert_close(actual,torch.tensor([1.,0.]))
        canvas[5] = 3
        torch.testing.assert_close(latent_information.decoded_prefix_weights(canvas,pos,3,14),torch.ones(2))

    def test_partition_retains_tail_mass_and_common_logit_invariance(self):
        logits = torch.tensor([[1.,2.,3.,4.]],dtype=torch.float64)
        tokens = torch.tensor([[0,3]])
        p = latent_information.partition_probabilities(logits,tokens)
        full = logits.float().softmax(-1)
        torch.testing.assert_close(p,torch.stack([full[:,0],full[:,3],full[:,1]+full[:,2]],-1))
        torch.testing.assert_close(p,latent_information.partition_probabilities(logits+20,tokens))

    def test_prefix_pool_stops_probes_when_decoded_prefix_is_filled(self):
        model = TinyModel(); prompt = torch.tensor([[1,2]])
        canvas = torch.tensor([[1,2,15,14,15,15,15,15]])
        result = latent_information.generate_information(model,prompt,torch.ones_like(prompt),
            initial_canvas=canvas,steps=3,gen_length=6,mask_id=15,eos_id=14,
            candidates=2,probe_every=1,branch_batch_size=4,target_vocab=4,
            prefix_targets=True,prefix_pool=True)
        self.assertNotEqual(int(result[0,2]),15)
        self.assertEqual(int(result[0,3]),14)
        self.assertEqual(latent_information.last_metrics['branch_work_by_row'],[2])

    def test_mi_defect_decomposition_and_shared_shift(self):
        conditional = torch.tensor([[[[.8,.1,.1]],[[.1,.7,.2]]]],dtype=torch.float64)
        rho = torch.tensor([[.7,.3]],dtype=torch.float64)
        baseline = torch.tensor([[[.2,.3,.5]]],dtype=torch.float64)
        info,defect = latent_information.response_information(conditional,rho,baseline)
        expected = ((conditional*(conditional.log()-baseline[:,None].log())).sum(-1)*rho[...,None]).sum(1)
        torch.testing.assert_close(info+defect,expected,atol=1e-12,rtol=1e-12)
        identical = conditional[:,:1].expand_as(conditional)
        info,defect = latent_information.response_information(identical,rho,baseline)
        self.assertLess(float(info.abs().max()),1e-12)
        self.assertGreater(float(defect.sum()),0.)

    def test_zero_probe_weights_recover_fisher_control(self):
        model = TinyModel(); prompt = torch.tensor([[1,2],[3,4]])
        params = dict(steps=4,gen_length=8,mask_id=15,eos_id=14,gain_beta=0.)
        expected = generate_finite(model,prompt,torch.ones_like(prompt),**params)
        actual = latent_information.generate_information(model,prompt,torch.ones_like(prompt),**params)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(latent_information.last_metrics['branch_work_by_row'],[0,0])

    def test_independent_positions_have_zero_external_information(self):
        model = TinyModel(coupled=False); prompt = torch.tensor([[1,2],[3,4]])
        actual = latent_information.generate_information(model,prompt,torch.ones_like(prompt),
            steps=3,gen_length=8,mask_id=15,eos_id=14,gain_beta=[0.,.5],
            candidates=2,probe_every=1,branch_batch_size=4,target_vocab=4)
        m = latent_information.last_metrics
        self.assertTrue(torch.equal(actual[:,:2],prompt))
        self.assertLess(abs(m['information_sum']),1e-6)
        self.assertLess(abs(m['consistency_defect_sum']),1e-6)
        self.assertEqual(m['branch_work_by_row'][0],0)
        self.assertGreater(m['branch_work_by_row'][1],0)
        self.assertEqual(m['branch_work_by_row'],m['probe_control_work_by_row'])
        self.assertEqual(sum(m['forward_work_by_row']),m['sequence_evaluations'])
        self.assertFalse(m['source_entropy_counted'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
