from types import SimpleNamespace

import unittest
import torch

from schedulers import latent_objective as scheduler


def direct_objective(z, risk, selected, noise=1., rho=.15, lg=1., lm=.15, ld=1.):
    k = noise*torch.eye(len(z), dtype=torch.double)+z.double()@z.double().T
    s = selected.nonzero().flatten()
    t = (~selected).nonzero().flatten()
    def ldmat(ids):
        return torch.linalg.slogdet(k[ids][:,ids])[1] if len(ids) else torch.tensor(0.)
    information = .5*(ldmat(s)+ldmat(t)-torch.linalg.slogdet(k)[1])
    dependence = .5*(k.diag()[s].log().sum()-ldmat(s))
    return rho*len(s)+lg*information-lm*risk[s].sum()-ld*dependence


def test_trace_matches_direct_objective_and_padding_is_ignored():
    generator = torch.Generator().manual_seed(19)
    for _ in range(20):
        z = torch.randn((3, 9, 4), generator=generator, dtype=torch.double)
        risk = torch.rand((3, 9), generator=generator, dtype=torch.double)
        active = torch.ones((3,9), dtype=torch.bool)
        active[1, 6:] = False
        active[2] = False
        chosen, detail = scheduler.gaussian_select(z, risk, active, noise=.7)
        assert not bool((chosen & ~active).any())
        for row in (0,1):
            expected = direct_objective(z[row,active[row]],risk[row,active[row]],
                                        chosen[row,active[row]],noise=.7)
            torch.testing.assert_close(detail['objective'][row],expected,rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(detail['contributions'].sum(-1),detail['objective'])
        assert detail['objective'][2] == 0


def test_independent_correlated_and_high_risk_counts():
    active = torch.ones((1,8), dtype=torch.bool)
    risk = torch.zeros((1,8))
    independent, _ = scheduler.gaussian_select(torch.zeros((1,8,1)),risk,active,
                                               risk_weight=1.,progress_reward=.05)
    correlated, detail = scheduler.gaussian_select(torch.full((1,8,1),2.),risk,active,
                                                  risk_weight=1.,progress_reward=.05)
    high_risk, fallback = scheduler.gaussian_select(torch.zeros((1,8,1)),risk+.3,active,
                                                   risk_weight=1.,progress_reward=.05)
    assert independent.sum() == 8
    assert correlated.sum() == 1
    assert not detail['forced'].item()
    assert high_risk.sum() == 1 and fallback['forced'].item()


def test_lowrank_features_preserve_gram_when_not_truncated():
    late = torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]])
    early = late.flip(1)
    active = torch.ones((1,3),dtype=torch.bool)
    full = scheduler.innovation_features(late,early,active,rank=4)
    reduced = scheduler.innovation_features(late,early,active,rank=2)
    # These features have matrix rank two, so truncation loses no Gram energy.
    torch.testing.assert_close(full@full.transpose(-1,-2),
                               reduced@reduced.transpose(-1,-2),rtol=1e-8,atol=1e-8)


def test_proposal_risk_uses_changes_rather_than_hidden_readout_levels():
    torch.manual_seed(29)
    final,middle,early,previous=[torch.randn(2,7,4) for _ in range(4)]
    directions=torch.randn(2,7,4)
    directions/=directions.norm(dim=-1,keepdim=True)
    active=torch.ones(2,7,dtype=torch.bool)
    shift=torch.randn(2,7,4)
    for measure in ('proposal_support','proposal_curvature'):
        before=scheduler.proposal_risk(final,middle,early,previous,directions,active,measure)
        after=scheduler.proposal_risk(final+shift,middle+shift,early+shift,
                                    previous+shift,directions,active,measure)
        torch.testing.assert_close(before,after)


def test_centered_covariance_excludes_shared_drift_and_padded_positions():
    torch.manual_seed(23)
    late,early=[torch.randn(2,7,4) for _ in range(2)]
    active=torch.ones(2,7,dtype=torch.bool)
    active[0,-2:]=False
    shifts=[torch.randn(2,1,4) for _ in range(2)]
    a=scheduler.innovation_features(late,early,active,rank=8,centered=True)
    b=scheduler.innovation_features(late+shifts[0],early+shifts[1],active,rank=8,centered=True)
    torch.testing.assert_close(a@a.transpose(-1,-2),b@b.transpose(-1,-2),rtol=1e-5,atol=1e-6)
    assert not bool(a[0,-2:].any())


class Block(torch.nn.Module):
    def forward(self,x):
        return (torch.tanh(x+.1),)


class Model(torch.nn.Module):
    def __init__(self,scale):
        super().__init__()
        torch.manual_seed(17)
        self.embedding = torch.nn.Embedding(8,2)
        self.head = torch.nn.Linear(2,8,bias=False)
        self.model = SimpleNamespace(transformer=SimpleNamespace(
            blocks=torch.nn.ModuleList([Block() for _ in range(5)]),ln_f=torch.nn.Identity()))
        self.scale = scale
        self.calls = 0

    def get_output_embeddings(self):
        return self.head

    def forward(self,ids,attention_mask):
        self.calls += 1
        hidden = self.embedding(ids)
        for block in self.model.transformer.blocks:
            hidden = block(hidden)[0]
        return SimpleNamespace(logits=self.head(hidden)*self.scale)


def test_generation_uses_one_forward_and_is_invariant_to_confidence_scale():
    prompt = torch.tensor([[2,3]])
    attention = torch.ones_like(prompt)
    results=[]
    for scale in (1.,100.):
        model=Model(scale)
        out=scheduler.generate_objective(model,prompt,attention,steps=4,gen_length=4,
            mask_id=6,projection=torch.eye(2),rank=2,incoming_weight=0.,
            risk_weight=0.,gain_weight=0.,dependence_weight=0.,max_count=2)
        assert torch.equal(out[:,:2],prompt)
        assert not bool((out[:,2:]==6).any())
        assert model.calls == scheduler.last_metrics['steps_taken'] == 2
        assert scheduler.last_metrics['sequence_evaluations'] == 2
        results.append(out)
    assert torch.equal(*results)


def test_temporal_only_bootstraps_then_batches_independent_stable_positions():
    model=Model(1.)
    prompt=torch.tensor([[2,3]])
    out=scheduler.generate_objective(model,prompt,torch.ones_like(prompt),steps=4,
        gen_length=4,mask_id=6,projection=torch.eye(2),rank=2,incoming_weight=0.,
        risk_measure='proposal_curvature',curvature_weight=0.,temporal_weight=1.,
        covariance_mode='risk_scaled',gain_weight=0.,risk_weight=.15,
        dependence_weight=1.,max_count=2)
    assert scheduler.last_metrics['reveal_counts']==[[1],[2],[1]]
    assert scheduler.last_metrics['bootstrap_singletons_by_row']==[1]
    assert model.calls==3
    assert not bool((out[:,2:]==6).any())


class ObjectiveChecks(unittest.TestCase):
    pass


for _name, _check in list(globals().items()):
    if _name.startswith('test_') and callable(_check):
        setattr(ObjectiveChecks,_name,lambda self,check=_check:check())
del _name, _check


if __name__ == '__main__':
    unittest.main()
