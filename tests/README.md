# Tests

Smoke and integration tests — run from the **repo root**:

| Test | Purpose |
|------|---------|
| `test_schedulers.py` | Scheduler decode smoke tests |
| `test_tracing.py` | Trace hooks across models × schedulers |

```bash
python tests/test_schedulers.py --scheduler pure_diffusion
python tests/test_tracing.py --model-path ./models/llada
./scripts/test_schedulers.sh --batch --batch_size 2
pytest tests/   # if pytest installed
```

`conftest.py` adds the repo root to `sys.path` for pytest.
