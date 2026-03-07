---
paths:
  - "miles/utils/ft/**/*.py"
---

## ft package architecture rule

Code outside `platform/` must NOT reference Ray or Kubernetes:
- No `import ray`, `from ray`, or `.remote()` calls
- No `import kubernetes` or `from kubernetes`
- Exception: `fault_injectors/` (test-only tooling, Ray usage allowed)
- Any Ray/K8s interaction must live in `platform/` and be exposed to other layers via Protocol interfaces
