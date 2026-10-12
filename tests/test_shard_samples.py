import json

from schedulerlens.shard_samples import sample_indices, write_samples_file


def test_shards_are_disjoint_and_cover_limited_task(tmp_path):
    first = sample_indices("humaneval", 2, 0, limit=9)
    second = sample_indices("humaneval", 2, 1, limit=9)

    assert set(first).isdisjoint(second)
    assert sorted(first + second) == list(range(9))

    output = write_samples_file(
        tmp_path / "samples.json",
        "humaneval",
        2,
        1,
        limit=9,
    )
    assert json.loads(output.read_text()) == {"humaneval": second}
