# Fault Tolerance (`miles.utils.ft`)

***WARN: This is experimental and can only be stabilized after sufficiently many large training runs.***

## Goals

These failures are considered:
1. Program error (recoverable by restarting corresponding components or the whole system)
2. Node error (only recoverable by removing the bad nodes and use replacement good nodes)

These constraints are considered:
* Be extensible: can add arbitrary complex detectors / diagnostics / sub state machines / … when we find sth is useful in large scale experiments
* Avoid unnecessary complexity / abstractions: we only need it for miles, not for all libraries in the world
* Avoid too strong assumptions to the environment: not assume users have Prometheus, DCGM exporter, node diagnoser, etc; only assume a Kubernetes + KubeRay; and this is in adapter layer thus replaceable

## Architecture & State Machines

### Figure

![](ft_arch.svg)

## Directory Structures

The directory layout also tightly follows the architecture.

- **Platform layer**: Platform-specific details, such as Kubernetes node-label adapter and Ray job adapter.
- **Controller**: Central control logic. When **detectors** observe issues, controller will let **recovery** manager to take over. The latter will try to restart, run **diagnostics**, notify humans, etc.
  - **Detectors**: Detect faults based on metrics.
  - **Recovery**: Multi-phase recovery state machine.
  - **Diagnostics**: On-demand diagnostics tools.
- **Agents**: Per-node and per-rank objects to collect metric and do actions.
  - **Collectors**: Collect various metrics.

## Naming Conventions

- **`Ft` prefix**: Only used on top-level entry classes that may be imported by external code (outside `miles.utils.ft`): `FtController`, `FtNodeAgent`, `FtControllerConfig`, `FtBaseModel`. Internal classes do not use the prefix.
- **Factory functions**: `build_*` for complex assembly with dependency injection; `create_*` for lightweight instantiation (e.g. state-machine steppers).
- **Timeout parameters**: Always `timeout_seconds` (never bare `timeout`).

## Tests

TODO: this will be outdated, to be updated

- `tests/fast/utils/ft/`
  - Others: Unit tests
  - `integration/in_process`: In-process integration tests
  - `integration/local_ray`: Integration tests based on local Ray
- `tests/e2e/ft/`: Realistic multi-node end-to-end tests

### Remote local-Ray testing notes

For `miles` development, FT tests should be executed remotely. For local-Ray based FT tests that start their own Ray cluster inside the container, prefer:

```bash
msc exec --no-ray --flavor miles_default --server ac-h200-n04 \
  'cd miles && /usr/bin/python3 -m pytest --confcutdir=tests/fast/utils/ft tests/fast/utils/ft/integration/...'
```

Why `--no-ray` matters:

- These tests already create a local Ray cluster inside the container.
- `msc exec` with Ray mode enabled performs a host-level cleanup using `pkill -9 -f session_`.
- On shared servers this cleanup may fail because unrelated root-owned Ray processes exist, causing the test run to fail before pytest starts.

The local multi-node Ray fixture in [`tests/fast/utils/ft/integration/conftest.py`](/Users/tom/Main/research/workspaces/ws_main/miles/tests/fast/utils/ft/integration/conftest.py) should keep the dashboard disabled (`--include-dashboard=false`). On the remote environment used for FT testing, enabling the Ray dashboard in the multi-node fixture was not reliable and could fail during `ray start` before the testbed even launched.

## Acknowledgements

TODO: papers websites etc
