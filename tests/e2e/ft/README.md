# FT E2E Tests

End-to-end tests for the fault-tolerance (FT) system. Tests run on a real Ray cluster with GPUs.

## Prerequisites

- A 3-node GPU Ray cluster (2 training nodes + 1 spare for eviction)
- `msc` CLI configured with flavor `miles_default`

## Running Tests

```bash
# Run all FT E2E tests
msc exec --flavor miles_default \
  'cd miles && /usr/bin/python3 -m pytest tests/e2e/ft/ -x --timeout=600'

# Run a specific test
msc exec --flavor miles_default \
  'cd miles && /usr/bin/python3 -m pytest tests/e2e/ft/test_transient_crash.py -x --timeout=600'
```

> Use `/usr/bin/python3 -m pytest` instead of `uv run pytest` — Ray workers break under `uv run`.

## Test Cases

| File | Scenario |
|------|----------|
| `test_transient_crash.py` | Single process kill → auto-recovery |
| `test_repeated_crash.py` | Multiple consecutive crashes → recovery |
| `test_hang.py` | Training process hang (SIGSTOP) → detection + recovery |
| `test_python_exception.py` | Injected Python exception → detection + recovery |
| `test_mfu_decline.py` | MFU performance decline → node eviction |
| `test_no_false_positive.py` | Normal training produces no false fault alerts |
| `test_cli_diagnostics.py` | CLI diagnostic commands work during training |
