import unittest

from traces.trace_checks import deduplicate_prompt_traces, duplicate_prompt_stats


def _sample(draw_idx: int) -> dict:
    return {
        "prompt": [[1, 2, 3]],
        "metadata": {"doc_id": 0, "draw_idx": draw_idx, "sample_idx": draw_idx},
    }


class StochasticTraceDedupTests(unittest.TestCase):
    def test_distinct_draws_of_same_prompt_are_preserved(self) -> None:
        samples = [_sample(0), _sample(1)]
        kept, repairs = deduplicate_prompt_traces(samples)

        self.assertEqual(len(kept), 2)
        self.assertEqual(repairs, [])
        self.assertEqual(duplicate_prompt_stats(samples)["removable_duplicates"], 0)

    def test_exact_duplicate_draw_is_removed(self) -> None:
        samples = [_sample(0), _sample(0)]
        kept, repairs = deduplicate_prompt_traces(samples)

        self.assertEqual(len(kept), 1)
        self.assertEqual(len(repairs), 1)


if __name__ == "__main__":
    unittest.main()
