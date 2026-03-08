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

## Error-as-Empty anti-pattern — FORBIDDEN on safety-critical paths

Never map "I/O error" or "infrastructure failure" to the same return value as "empty result" on paths where the caller makes a safety decision (fault detection, health checks, recovery logic, diagnostics).

Concretely, do NOT:
- `except Exception: return []` / `return None` / `return set()` / `return EMPTY_DF` when the caller interprets empty as "all clear"
- Return the same type/value for "confirmed no data" and "failed to fetch data"
- Use `graceful_degrade(default=...)` on functions whose callers treat the default as a safe/healthy signal

Instead:
- Let exceptions propagate to a layer that can handle them (e.g. the detector orchestrator, the recovery stepper)
- If a function must not raise, use a result type that distinguishes success-empty from error (e.g. `RetryResult`, a dataclass with an `ok` field, or `Optional` where `None` means error and empty collection means no-data)
- When catching exceptions for graceful degradation, ensure the caller can still distinguish "degraded" from "confirmed safe"

Rationale: this codebase is a fault-tolerance controller. "I don't know" must never be treated as "everything is fine." See `adhoc/2026-03-08-prometheus-error-propagation-plan.md` for the full audit.
