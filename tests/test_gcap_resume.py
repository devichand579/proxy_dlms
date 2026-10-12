import json

from eval import _load_checkpoint_completed


def test_resume_requires_every_gcap_candidate_and_step_data(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    complete_candidate = {"response": "answer", "step_proxies": [{"step": 0}]}
    rows = [
        {
            "task_name": "toy",
            "doc_id": 0,
            "resps": ["selected"],
            "gcap_rerank": {"candidates": [complete_candidate] * 8},
        },
        {
            "task_name": "toy",
            "doc_id": 1,
            "resps": ["selected"],
            "gcap_rerank": {"candidates": [complete_candidate] * 7},
        },
        {
            "task_name": "toy",
            "doc_id": 2,
            "resps": ["selected"],
            "gcap_rerank": {
                "candidates": [
                    complete_candidate,
                    *[
                        {"response": "answer", "step_proxies": []}
                        for _ in range(7)
                    ],
                ]
            },
        },
    ]
    checkpoint.write_text("".join(json.dumps(row) + "\n" for row in rows))

    completed = _load_checkpoint_completed(
        str(checkpoint),
        expected_gcap_k=8,
    )

    assert completed == {("toy", 0): ["selected"]}
