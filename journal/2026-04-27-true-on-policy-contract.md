---
date: 2026-04-27
operator: codex
work_type: migration
repo: miles
branch: maocheng/top-dense-pr03-miles-wiring
commit: e9a4c27a5587be778bf2af97d31d84549deae86a
environment: ion-user-8 / miles-maocheng
status: success
tags: [true-on-policy, qwen3-dense, refactor]
---

# True-on-policy Miles contract refactor

## Objective

Move Qwen3-dense true-on-policy launch wiring toward one public knob by
centralizing derived SGLang/Megatron args and env vars in `miles.true_on_policy`.
This slice preserves the existing run behavior.

## Environment

- **Local code**: `recovery/qwen3_dense_clean/miles`
- **Remote host**: `radixark@ion-user-8.tail134ba0.ts.net`
- **Container**: `miles-maocheng`
- **Remote path**: `/root/miles`
- **Synced files**:
  - `scripts/run_qwen3_4b.py`
  - `miles/true_on_policy/`
  - `tests/fast/true_on_policy/`

## Commands

```bash
tar -czf /tmp/miles-top-contract-slice.tgz \
  scripts/run_qwen3_4b.py miles/true_on_policy tests/fast/true_on_policy
scp /tmp/miles-top-contract-slice.tgz \
  radixark@ion-user-8.tail134ba0.ts.net:/tmp/miles-top-contract-slice.tgz
ssh radixark@ion-user-8.tail134ba0.ts.net \
  "docker cp /tmp/miles-top-contract-slice.tgz miles-maocheng:/tmp/miles-top-contract-slice.tgz && \
   docker exec -w /root/miles miles-maocheng bash -lc \
   'tar -xzf /tmp/miles-top-contract-slice.tgz -C /root/miles'"
```

```bash
ssh radixark@ion-user-8.tail134ba0.ts.net \
  "docker exec -w /root/miles miles-maocheng bash -lc \
   'pytest -q tests/fast/true_on_policy/test_config.py tests/fast/true_on_policy/test_run_qwen3_4b.py'"
```

```bash
ssh radixark@ion-user-8.tail134ba0.ts.net \
  "docker exec -w /root/miles miles-maocheng bash -lc \
   'python -m compileall -q miles/true_on_policy tests/fast/true_on_policy scripts/run_qwen3_4b.py && echo compileall-ok'"
```

## Result

- Focused remote pytest: `11 passed, 3 warnings in 0.05s`
- Remote compile check: `compileall-ok`
- Local direct import/assert checks passed.
- Local pytest was not used as the primary signal because it hung during startup in
  this local environment.

## Interpretation

The first refactor slice is behavior-preserving for Qwen3-dense launch wiring:
`--true-on-policy` still expands to the same SGLang deterministic args, Megatron
SGLang-compatible args, and TP-invariant env vars when required.

## Follow-ups

- [ ] Split SGLang true-on-policy helper/kernels into a clearer module boundary.
- [ ] Split Megatron SGLang-compatible backend pieces into a clearer module boundary.
- [ ] Add distributed topology tests for TP/CP mismatch and backward grad coverage.
