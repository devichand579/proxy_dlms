import unittest
import torch
from schedulerlens.context_reconstruction_audit import masked_context_score,selection_report
from test_latent_scheduler import TinyModel


class ReconstructionChecks(unittest.TestCase):
    def test_independent_model_cannot_score_body_semantics(self):
        model=TinyModel(coupled=False)
        inputs=torch.tensor([[15,15,1,2],[15,15,7,8]])
        reference=torch.tensor([[3,4,1,2],[3,4,7,8]])
        targets=torch.tensor([[True,True,False,False]]).repeat(2,1)
        score,count=masked_context_score(model,inputs,torch.ones_like(inputs),reference,targets,15)
        torch.testing.assert_close(score[:1],score[1:]); self.assertEqual(count.tolist(),[2,2])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_visible_targets_are_rejected(self):
        model=TinyModel(); inputs=torch.tensor([[1,15]])
        with self.assertRaises(ValueError):
            masked_context_score(model,inputs,torch.ones_like(inputs),inputs,
                                 torch.tensor([[True,False]]),15)

    def test_selection_never_uses_correctness_to_choose(self):
        candidates=[dict(task_id='x',passed=True,score=-4.,sources=['good']),
                    dict(task_id='x',passed=False,score=-2.,sources=['bad'])]
        report=selection_report(candidates,'score')
        self.assertEqual(report['selected_passed'],0)
        self.assertEqual(report['selected'][0]['sources'],['bad'])
        self.assertEqual(report['available_correct_tasks'],1)
