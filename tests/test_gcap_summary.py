import json
from types import SimpleNamespace

from schedulerlens import gcap_summary


def test_summary_computes_exact_pass1_and_prefix_selection(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "task_name": "toy",
            "doc_id": doc_id,
            "gcap_rerank": {
                "candidates": [
                    {
                        "response": first,
                        "prefix_steps": 32,
                        "selection_score": first_score,
                    },
                    {
                        "response": second,
                        "prefix_steps": 32,
                        "selection_score": second_score,
                    },
                ]
            },
        }
        for doc_id, first, second, first_score, second_score in (
            (0, "0", "1", 2.0, 1.0),
            (1, "1", "0", 1.0, 0.5),
        )
    ]
    (run_dir / "checkpoint.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    fake_task = SimpleNamespace(
        instances=[SimpleNamespace(doc_id=0), SimpleNamespace(doc_id=1)]
    )
    monkeypatch.setattr(gcap_summary, "_load_scoring_task", lambda *args: fake_task)
    monkeypatch.setattr(gcap_summary, "_is_code_task", lambda task: False)
    monkeypatch.setattr(
        gcap_summary,
        "_metric_value",
        lambda task, instance, task_name, response: float(response),
    )

    result = gcap_summary.summarize(
        task_name="toy",
        scheduler="pd",
        run_dirs=[run_dir],
        prefix_steps=32,
        k=2,
    )

    assert result["pass_at_1"] == 0.5
    assert result["pass_at_8"] == 1.0
    assert result["min_g_prefix_selected_at_1"] == 0.5
