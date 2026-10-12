from schedulerlens.proxies import SampleProxies, StepProxies
from schedulers import gcap_rerank


def test_candidate_metrics_keep_every_step_and_select_first_prefix(monkeypatch):
    steps = [
        StepProxies(
            step=0,
            M_hat_t=1.0,
            J_hat_t=2.0,
            n_committed=2,
            n_selected=2,
            n_remaining=510,
        ),
        StepProxies(
            step=31,
            M_hat_t=3.0,
            J_hat_t=4.0,
            n_committed=3,
            n_selected=1,
            n_remaining=509,
        ),
        StepProxies(
            step=32,
            M_hat_t=10.0,
            J_hat_t=20.0,
            n_committed=5,
            n_selected=2,
            n_remaining=507,
        ),
    ]
    proxies = SampleProxies(
        M_hat=14.0,
        J_hat=26.0,
        step_proxies=steps,
    )
    monkeypatch.setattr(gcap_rerank, "repair_drop_last_step", lambda trace: (trace, []))
    monkeypatch.setattr(
        gcap_rerank,
        "compute_sample_proxies",
        lambda trace: proxies,
    )

    metrics = gcap_rerank._candidate_metrics(
        {"steps": []},
        "per_commit",
        selection_prefix_steps=32,
    )

    assert metrics["M_hat"] == 14.0
    assert metrics["J_hat"] == 26.0
    assert metrics["prefix_M_hat"] == 4.0
    assert metrics["prefix_J_hat"] == 6.0
    assert metrics["prefix_G_hat"] == 10.0
    assert metrics["prefix_n_committed"] == 3
    assert metrics["selection_score"] == 10.0 / 3.0
    assert [row["step"] for row in metrics["step_proxies"]] == [0, 31, 32]
    assert [row["G_hat_t"] for row in metrics["step_proxies"]] == [3.0, 7.0, 30.0]
    assert [row["cumulative_G_hat"] for row in metrics["step_proxies"]] == [
        3.0,
        10.0,
        40.0,
    ]
