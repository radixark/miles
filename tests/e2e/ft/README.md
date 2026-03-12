# FT E2E Tests

End-to-end tests for the fault-tolerance (FT) system. Tests run on a real Ray cluster with GPUs.

## Adding Tests

Most e2e tests should also have a counterpart in `local_ray_semi_e2e`.
Therefore, please:

1. Define scenarios inside `tests/fast/utils/ft/integration/local_ray_semi_e2e/scenarios`
2. Add tests in `tests/fast/utils/ft/integration/local_ray_semi_e2e`
2. Add tests here (`tests/e2e/ft`)

## Prerequisites

- A 4-node GPU Ray cluster (3 nodes for disaggregated rollout and training + 1 spare for eviction)

## Running Tests

```bash
msc exec --flavor miles_ft_e2e_test 'cd miles && /usr/bin/python3 -m pytest tests/e2e/ft/ -x'
```

> Use `/usr/bin/python3 -m pytest` instead of `uv run pytest` — Ray workers break under `uv run`.
