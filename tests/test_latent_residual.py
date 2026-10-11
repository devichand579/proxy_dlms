import copy
import itertools
from types import SimpleNamespace
import unittest

import torch

from schedulers import latent_residual


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8,8,bias=False)

    def forward(self,x):
        return x+.1*torch.tanh(self.linear(x+x.mean(1,keepdim=True))),None


class ActivationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(17)
        self.embedding = torch.nn.Embedding(16,8)
        self.head = torch.nn.Linear(8,16)
        with torch.no_grad(): self.head.bias[14:] = -100
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.blocks = torch.nn.ModuleList([ToyBlock() for _ in range(5)])
        self.model.transformer.ln_f = torch.nn.LayerNorm(8)
        self.logit_scale = 1.

    def get_output_embeddings(self): return self.head

    def forward(self,ids,attention_mask=None):
        h = self.embedding(ids)
        for block in self.model.transformer.blocks: h,_ = block(h)
        h = self.model.transformer.ln_f(h)
        return SimpleNamespace(logits=self.head(h)*self.logit_scale)


class ResidualTests(unittest.TestCase):
    def test_spectral_solver_matches_exhaustive_objective_and_count(self):
        torch.manual_seed(4)
        response = torch.randn(5,7,dtype=torch.double)
        interaction = 2*response@response.T
        gain = torch.tensor([4.,2.,3.,1.,2.],dtype=torch.double)
        selected,diag = latent_residual.subset_optimum(gain,interaction)
        patterns = torch.tensor(list(itertools.product([0.,1.],repeat=5)),dtype=torch.double)
        objective = patterns@gain-.5*(patterns@interaction*patterns).sum(-1)
        v = torch.zeros(5,dtype=torch.double); v[selected] = 1
        value = float(v@gain-.5*v@interaction@v)
        self.assertAlmostEqual(value,float(objective.max()),places=8)
        self.assertFalse(diag['forced_singleton'])

    def test_interference_changes_count_without_changing_individual_gains(self):
        gain = torch.ones(4)
        independent,_ = latent_residual.subset_optimum(gain,torch.eye(4))
        aligned,_ = latent_residual.subset_optimum(gain,torch.ones(4,4))
        self.assertEqual(len(independent),4)
        self.assertEqual(len(aligned),1)

    def test_negative_gain_forces_only_one_progress_commit(self):
        chosen,diag = latent_residual.subset_optimum(torch.tensor([-1.,-.1,-2.]),torch.eye(3))
        self.assertEqual(chosen.tolist(),[1])
        self.assertTrue(diag['forced_singleton'])

    def test_surviving_targets_match_explicit_subset_residual_reduction(self):
        torch.manual_seed(5)
        response = torch.randn(3,5,4,dtype=torch.double)*.1
        field = torch.randn(5,4,dtype=torch.double)
        modular = torch.tensor([.1,.03,-.08],dtype=torch.double)
        source_targets = torch.tensor([0,2,4])
        chosen,diag = latent_residual.surviving_optimum(modular,response,field,source_targets)
        values = []
        for bits in itertools.product([0.,1.],repeat=3):
            v = torch.tensor(bits,dtype=torch.double)
            remaining = torch.ones(5,dtype=torch.bool)
            remaining[source_targets[v.bool()]] = False
            after = field-torch.einsum('i,ijd->jd',v,response)
            reduction = ((field[remaining].square()-after[remaining].square()).sum()/field.square().sum())
            values.append(float(reduction+v@modular))
        self.assertFalse(diag['forced_singleton'])
        self.assertAlmostEqual(diag['objective'],max(values),places=8)

    def test_revealed_source_cannot_claim_its_own_residual_reduction(self):
        response = torch.zeros(1,2,2,dtype=torch.double);response[0,0] = 1.
        field = torch.ones(2,2,dtype=torch.double)
        _,diag = latent_residual.surviving_optimum(torch.tensor([-.01]),response,field,torch.tensor([0]))
        self.assertTrue(diag['forced_singleton'])
        self.assertAlmostEqual(diag['objective'],-.01,places=8)

    def test_spectral_pool_covers_distinct_residual_axes(self):
        residual = torch.tensor([[1.,0.],[1.,0.],[-1.,0.],[-1.,0.],[0.,2.],[0.,-2.]])
        chosen = latent_residual.spectral_pool(residual,torch.ones(6),2)
        self.assertEqual(len(set(chosen.tolist())),2)
        selected = residual[chosen]
        self.assertAlmostEqual(float(selected[0]@selected[1]),0.,places=7)

    def test_latent_residual_reduction_identity_including_negative_cross_terms(self):
        response = torch.tensor([[1.,0.],[-.8,.3],[0.,.5]],dtype=torch.double)
        field = torch.tensor([.4,.8],dtype=torch.double)
        gain = 2*response@field
        interaction = 2*response@response.T
        for bits in itertools.product([0.,1.],repeat=3):
            v = torch.tensor(bits,dtype=torch.double)
            quadratic = v@gain-.5*v@interaction@v
            actual = field.square().sum()-(field-v@response).square().sum()
            self.assertAlmostEqual(float(quadratic),float(actual),places=9)

    def test_confidence_changes_leave_selection_count_and_completions_identical(self):
        model = ActivationModel(); scaled = copy.deepcopy(model); scaled.logit_scale = .07
        prompt = torch.tensor([[1,2],[3,4]])
        params = dict(steps=4,gen_length=5,mask_id=15,eos_id=14,candidates=2,
                      branch_batch_size=3,progress_reward=.75)
        original = latent_residual.generate_residual(model,prompt,torch.ones_like(prompt),**params)
        original_counts = latent_residual.last_metrics['reveal_counts']
        actual = latent_residual.generate_residual(scaled,prompt,torch.ones_like(prompt),**params)
        self.assertTrue(torch.equal(original,actual))
        self.assertEqual(original_counts,latent_residual.last_metrics['reveal_counts'])
        self.assertTrue(torch.equal(actual[:,:2],prompt))
        metrics = latent_residual.last_metrics
        self.assertEqual(sum(metrics['forward_work_by_row']),metrics['sequence_evaluations'])
        self.assertEqual(metrics['branch_work_by_row'],metrics['probe_control_work_by_row'])
        self.assertFalse(metrics['confidence_scoring'])
        self.assertFalse(metrics['entropy_scoring'])
        self.assertFalse(metrics['eb_count'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__': unittest.main()
