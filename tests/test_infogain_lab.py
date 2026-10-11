"""Scientific invariants for the small information-gain audit."""
import unittest

import numpy as np

from schedulerlens.infogain_lab import (
    binary_joint, dependence_decomposition, effect_metrics, entropy,
    policy_scores, total_correlation, toy_records,
)


class InformationGainChecks(unittest.TestCase):
    def test_removing_independent_source_does_not_inform_survivors(self):
        before = [entropy([.5, .5]), entropy([.95, .05])]
        external, state_gain = effect_metrics(before, before, 0)
        self.assertEqual(external, 0)
        self.assertGreater(state_gain, 0)

    def test_parity_has_zero_pairwise_dependence_but_nonzero_batch_tc(self):
        states, mass = binary_joint("parity")
        for pair in ((0, 1), (0, 2), (1, 2)):
            self.assertAlmostEqual(total_correlation(states, mass, pair), 0, places=12)
        self.assertAlmostEqual(total_correlation(states, mass, (0, 1, 2)), np.log(2), places=12)

    def test_branch_influence_separates_dependence_and_coherence_defect(self):
        influence, dependence, defect = dependence_decomposition([.5, .5], [[.9, .1], [.7, .3]], [.25, .75])
        self.assertGreater(dependence, 0)
        self.assertGreater(defect, 0)
        self.assertAlmostEqual(influence, dependence + defect, places=12)

    def test_sharper_wrong_predictions_can_harm_reference_likelihood(self):
        _, checks = toy_records()
        self.assertGreater(checks["wrong_anchor"]["external_entropy_gain"], 0)
        self.assertLess(checks["wrong_anchor"]["remaining_reference_nll_gain"], 0)

    def test_reference_labels_never_enter_selection_scores(self):
        records, _ = toy_records()
        changed = [{**r, "reference_nll_gain": -1e6, "source_error": 1} for r in records]
        for name in ("confidence", "margin", "entropy", "paper_ig", "external_gain", "response_kl"):
            self.assertEqual(policy_scores(records, name, .25), policy_scores(changed, name, .25))

    def test_exact_coherent_model_has_no_marginal_consistency_defect(self):
        rows, _ = toy_records()
        self.assertLess(max(abs(row["coherence_defect"]) for row in rows), 1e-12)
        independent = [r for r in rows if r["sample_id"] == "independent"]
        self.assertLess(max(abs(r["external_gain"]) for r in independent), 1e-12)


if __name__ == "__main__":
    unittest.main()
