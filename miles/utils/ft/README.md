# Fault Tolerance (`miles.utils.ft`)

***WARN: This is experimental and can only be stabilized after sufficiently many large training runs.***

## Architecture

The directory layout also tightly follows the architecture.

- **Platform layer**: Platform-specific details, such as Kubernetes node-label adapter and Ray job adapter.
- **Controller**: Central control logic. When **detectors** observe issues, controller will let **recovery** manager to take over. The latter will try to restart, run **diagnostics**, notify humans, etc.
  - **Detectors**: Detect faults based on metrics.
  - **Recovery**: Multi-phase recovery state machine.
  - **Diagnostics**: On-demand diagnostics tools.
- **Agents**: Per-node and per-rank objects to collect metric and do actions.
  - **Collectors**: Collect various metrics.

## Tests

TODO: this will be outdated, to be updated

- `tests/fast/utils/ft/`
  - Others: Unit tests
  - `integration/in_process`: In-process integration tests
  - `integration/local_ray`: Integration tests based on local Ray
- `tests/e2e/ft/`: Realistic multi-node end-to-end tests
