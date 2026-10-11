import copy
import unittest

import torch

from schedulers import latent_allowance as lab
from test_latent_count import model


class AllowanceChecks(unittest.TestCase):
    def test_only_committed_attended_end_markers_bound_candidates(self):
        tokens=torch.tensor([[15,2,12,15,15],[15,15,15,12,12]])
        attention=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
        active,ends=lab.committed_end_region(tokens,attention,15,12)
        self.assertEqual(ends.tolist(),[2,5])
        self.assertEqual(active.tolist(),[[True,False,False,False,False],[True,True,True,False,False]])
    def test_independent_and_coupled_prefixes(self):
        order=torch.arange(4);error=torch.zeros(4)
        independent,info=lab.allowance_prefix(order,error,torch.zeros(4,4),.01)
        coupled,_=lab.allowance_prefix(order,error,torch.ones(4,4)*.5,.01)
        self.assertEqual(len(independent),4)
        self.assertEqual(len(coupled),1)
        self.assertEqual(info['cost'],0)

    def test_additive_and_pair_costs_and_recorded_override(self):
        order=torch.arange(3);error=torch.tensor([.02,.03,.1])
        attention=torch.tensor([[0.,.02,.04],[.02,0.,.01],[.04,.01,0.]])
        chosen,info=lab.allowance_prefix(order,error,attention,.08)
        self.assertEqual(chosen.tolist(),[0,1])
        expected=.05+.02*((1.02*1.03)**.5)
        self.assertAlmostEqual(info['cost'],expected,places=6)
        self.assertFalse(info['forced'])
        chosen,info=lab.allowance_prefix(order,error,attention,.001)
        self.assertEqual(chosen.tolist(),[0])
        self.assertTrue(info['forced'])
        self.assertGreater(info['cost'],.001)

    def test_subset_skips_costly_interaction_and_preserves_anchor(self):
        order=torch.arange(3);error=torch.zeros(3);attention=torch.zeros(3,3)
        attention[0,1]=attention[1,0]=.5
        chosen,diag=lab.allowance_subset(order,error,attention,.01)
        self.assertEqual(chosen.tolist(),[0,2])
        self.assertEqual(diag['skipped'],[1])
        self.assertFalse(diag['forced'])

    def test_feedback_shrinks_and_limits_growth(self):
        self.assertEqual(lab.feedback_count(4,4.,.01,100),1)
        self.assertEqual(lab.feedback_count(4,0.,.01,100),8)
        self.assertEqual(lab.feedback_count(4,0.,.01,3),3)
        self.assertEqual(lab.feedback_count(16,0.,.01,100),16)
        self.assertEqual(lab.feedback_count(2,.1,.5,100),
                         lab.feedback_count(2,10.,50.,100))

    def test_revealed_sources_never_count_as_external_response(self):
        error=torch.tensor([100.,.01,.01]);active=torch.tensor([False,True,True])
        self.assertAlmostEqual(lab.response_density(error,active),.01,places=6)
        attention=torch.eye(3)
        attention.diagonal().zero_()
        self.assertEqual(lab.response_density(error,active,attention=attention),0.)

    def test_gradient_metric_head_gauge_scale_and_duplicate_invariance(self):
        torch.manual_seed(31)
        weight=torch.randn(9,5);current=torch.randn(2,3,5)
        previous=torch.randn(2,3,5);native=torch.randn(2,3,5)
        ids=torch.tensor([[[1,2],[2,3],[3,4]],[[4,5],[5,6],[6,7]]])
        active=torch.ones(2,3,dtype=torch.bool)
        a=lab.gradient_ratio(weight,current,previous,native,ids,ids,active)
        b=lab.gradient_ratio(7*weight+torch.randn(1,5),current,previous,native,ids,ids,active)
        c=lab.gradient_ratio(weight,current,previous,native,ids.repeat(1,1,2),ids,active)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(a,c,rtol=1e-5,atol=1e-5)

    def test_fisher_diagnostic_distinguishes_geometry_at_equal_entropy(self):
        logits=torch.tensor([[[0.,0.,-100.,-100.],[-100.,-100.,0.,0.]]])
        dictionary=torch.tensor([[0.,0.],[.1,0.],[-2.,0.],[2.,0.]])
        dictionary-=dictionary.mean(0)
        value=lab.gradient_dispersion(logits,dictionary)
        self.assertGreater(float(value[0,1]),100*float(value[0,0]))
        certain=torch.full_like(logits,-100.);certain[:,:,0]=0
        self.assertLess(float(lab.gradient_dispersion(certain,dictionary).max()),1e-6)

    def test_one_forward_per_step_prefix_and_bootstrap(self):
        base=model();prompt=torch.tensor([[2,3],[4,5]])
        for controller in ('feedback','feedback_grounded','feedback_relative','feedback_relative_grounded',
                           'allowance','allowance_grounded','allowance_subset'):
            value=copy.deepcopy(base);calls=[]
            hook=value.register_forward_hook(lambda *_:calls.append(1))
            out=lab.generate_allowance(value,prompt,torch.ones_like(prompt),steps=4,
                gen_length=6,mask_id=15,ranking='activation',controller=controller,
                projection=torch.eye(8),metric='gradient_union',tolerance=.025)
            hook.remove()
            self.assertTrue(torch.equal(out[:,:2],prompt))
            self.assertEqual(len(calls),lab.last_metrics['steps_taken'])
            self.assertEqual(lab.last_metrics['sequence_evaluations'],2*len(calls))
            self.assertEqual(lab.last_metrics['reveal_counts'][0],[1,1])
            self.assertFalse(lab.last_metrics['confidence_scoring'])
            self.assertFalse(lab.last_metrics['eb_count'])
            self.assertFalse(lab.last_metrics['count_uses_vocabulary_probabilities'])

    def test_end_completion_stops_without_reading_hidden_answers(self):
        value=model()
        with torch.no_grad():value.head.bias[12]=100.
        prompt=torch.tensor([[2,3]])
        result=lab.generate_allowance(value,prompt,torch.ones_like(prompt),steps=4,
            gen_length=6,mask_id=15,eos_id=12,ranking='diagnostic_fisher',controller='singleton',
            projection=torch.eye(8),metric='uniform',candidate_region='committed_eos')
        self.assertEqual(lab.last_metrics['steps_taken'],1)
        self.assertEqual(lab.last_metrics['logical_eos_completion_by_row'],[1])
        self.assertEqual(int(result[0,2]),12)


if __name__=='__main__':unittest.main()
