# FT E2E Tests

End-to-end tests for the fault-tolerance (FT) system. Tests run on a real Ray cluster with GPUs.

## Prerequisites

- A 4-node GPU Ray cluster (3 nodes for disaggregated rollout and training + 1 spare for eviction)

## Running Tests

```bash
msc exec --flavor miles_ft_e2e_test 'cd miles && /usr/bin/python3 -m pytest tests/e2e/ft/ -x'
```

> Use `/usr/bin/python3 -m pytest` instead of `uv run pytest` — Ray workers break under `uv run`.
