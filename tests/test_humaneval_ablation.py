import pytest

from schedulerlens.humaneval_ablation import (
    _selection_summary,
    pass_at_k,
    pearson_corr,
    roc_auc_for_lower_score,
    scheduler_from_run_name,
)


def test_pass_at_k_matches_extremes():
    assert pass_at_k(24, 0, 12) == 0.0
    assert pass_at_k(24, 24, 1) == 1.0
    assert pass_at_k(24, 1, 24) == 1.0
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)


def test_lower_proxy_auc_and_correlation_detect_signal():
    rows = [
        {"passed": True, "G_hat": 0.1},
        {"passed": True, "G_hat": 0.2},
        {"passed": False, "G_hat": 0.8},
        {"passed": False, "G_hat": 0.9},
    ]
    assert roc_auc_for_lower_score(rows, "G_hat") == 1.0
    assert pearson_corr(
        [row["G_hat"] for row in rows],
        [float(row["passed"]) for row in rows],
    ) < 0


def test_selection_summary_reports_top_pool_quality():
    by_doc = {
        0: [
            {"draw_idx": 0, "G_hat": 0.1, "passed": True},
            {"draw_idx": 1, "G_hat": 0.2, "passed": True},
            {"draw_idx": 2, "G_hat": 0.8, "passed": False},
            {"draw_idx": 3, "G_hat": 0.9, "passed": False},
        ]
    }
    summary = _selection_summary(by_doc, score="G_hat", subset_sizes=[2])
    assert summary["top1_pass_at_1"] == 1.0
    selected = summary["subset_sizes"]["2"]
    assert selected["top_pool_expected_pass_at_1"] == 1.0
    assert selected["top_pool_pass_at_m"] == 1.0
    assert selected["random_pool_expected_pass_at_1"] == 0.5


def test_two_node_shards_share_scheduler_key():
    shard0 = (
        "llada_humaneval_top_k_confidence_s512_t0p3_k24_"
        "shard0of2_fp32_none"
    )
    shard1 = (
        "llada_humaneval_top_k_confidence_s512_t0p3_k24_"
        "shard1of2_fp32_none"
    )
    assert scheduler_from_run_name(shard0) == "top_k_confidence"
    assert scheduler_from_run_name(shard1) == "top_k_confidence"
