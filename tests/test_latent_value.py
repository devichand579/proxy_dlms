import unittest
import tempfile
from pathlib import Path
import torch

from schedulerlens.train_latent_value import surviving_gain
from schedulers.latent_value import value_features, ValueReadout
from schedulers.latent_risk import RiskHead


class RevealValueChecks(unittest.TestCase):
    def test_source_improvement_does_not_count_as_external_gain(self):
        baseline = torch.tensor([-10., -2., -3.])
        positions = torch.tensor([4, 5, 6])
        self.assertEqual(float(surviving_gain(torch.tensor([0., -2., -3.]), baseline, positions, 4)), 0.)
        self.assertEqual(float(surviving_gain(torch.tensor([0., -1., -2.]), baseline, positions, 4)), 1.)
        self.assertEqual(float(surviving_gain(torch.tensor([0.]), baseline[:1], positions[:1], 4)), 0.)

    def test_target_context_feature_excludes_selected_source(self):
        hp = torch.eye(2); ep = torch.empty(2, 0)
        hidden = torch.tensor([[100., 200.]])
        other = torch.tensor([[1., 2.], [3., 4.]])
        x = value_features(hidden, torch.tensor([[0., 1., 2.]]), torch.zeros(3, 2),
                           hp, ep, 2, torch.cat([hidden, other]))
        torch.testing.assert_close(x[:, -2:], other.mean(0, keepdim=True))

    def test_scalar_control_ignores_hidden_state_changes(self):
        head = RiskHead(4)
        for parameter in head.parameters(): parameter.data.zero_()
        head.network[0].weight.data[0,0] = 1
        head.network[2].weight.data[0,0] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'value.pt'
            torch.save({'head':head.state_dict(), 'mean':torch.zeros(4), 'scale':torch.ones(4),
                'h_projection':torch.eye(2), 'e_projection':torch.empty(2,0),
                'feature_indices':torch.arange(2,6), 'metadata':{}},path)
            readout = ValueReadout(path,device='cpu')
            logits = torch.tensor([[0.,1.,2.]])
            embeddings = torch.zeros(3,2)
            a = readout.predict(torch.ones(1,2),logits,embeddings,2,torch.ones(3,2))
            b = readout.predict(torch.full((1,2),100.),logits,embeddings,2,torch.full((3,2),200.))
            expected = torch.nn.functional.silu(-logits.log_softmax(-1).amax(-1))
            torch.testing.assert_close(a,expected)
            torch.testing.assert_close(a,b)


if __name__ == '__main__': unittest.main()
