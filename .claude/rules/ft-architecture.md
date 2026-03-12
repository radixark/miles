---
paths:
  - "miles/utils/ft/**/*.py"
  - "tests/fast/utils/ft/testbed/**/*.py"
---

## ft architecture rules

### Adapters

* **No Ray/K8s outside adapters**: Code outside `adapters/` must NOT reference Ray or Kubernetes (`import ray`, `.remote()`, `import kubernetes`). Exception: `fault_injectors/` (test-only).
* **Adapter actors should be thin**: only transport-layer conversion (`.remote()` / `ray.get()`). Business logic belongs in the core layer (`controller/`, `agents/`).

### Layer dependencies

From top to bottom: `cli` > `factories` > `adapters` > `controller`, `agents` > `utils`

Each layer may only import from layers below it.
Exception: `controller` and `agents` may import `adapters/types.py` (the boundary contract — cross-layer protocols and constants).
`controller` and `agents` are peers and may import each other's type definitions.

### Error-as-Empty — FORBIDDEN on safety-critical paths

On fault detection / recovery / diagnostic paths, "I don't know" must never look like "everything is fine."
Do NOT catch exceptions and return empty (`[]`, `None`, `set()`) when callers interpret empty as "all clear."

### MilesTestbed — 1:1 alignment with miles package

`tests/fast/utils/ft/testbed/` mirrors the miles package directory structure 1:1. When adding or modifying testbed files:

* **File and directory names must match**: `testbed/ray/rollout.py` ← `miles/ray/rollout.py`, `testbed/backends/sglang_utils/sglang_engine.py` ← `miles/backends/sglang_utils/sglang_engine.py`, etc.
* **Class names use `Testbed` prefix**: `RolloutManager` → `TestbedRolloutManager`, `TrainRayActor` → `TestbedTrainRayActor`, etc.
* **Two mapping scopes**: `testbed/` mirrors `miles/` (training system), `testbed/utils/ft/` mirrors `miles/utils/ft/` (FT adapters).
* Design doc: `docs/13-enhance-tests/3-fake-miles-for-semi-e2e.md` (in rl_resilience repo) has the full mapping table.
