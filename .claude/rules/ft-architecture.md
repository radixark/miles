---
paths:
  - "miles/utils/ft/**/*.py"
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
