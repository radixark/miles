# [AMD] Seed the ROCm e2e CI suite with verified MI350X tests

## What this does

Registers the miles e2e tests that have been verified to pass end-to-end on
AMD MI350X (gfx950) into the `HWBackend.ROCM` CI suite, by adding a
`register_rocm_ci(...)` marker next to each test's existing `register_cuda_ci(...)`.
Each ROCm registration mirrors the CUDA one (same `est_time`, same `labels`) and
points at the matching `stage-c-*-mi350` suite.

The `register_rocm_ci` + `HWBackend.ROCM` framework lives in
`tests/ci/ci_register.py` and is parsed by the same AST machinery as
`register_cuda_ci` (runtime no-op marker). This PR grows the suite one
*individually verified* test at a time — nothing is registered that has not
actually run green on the hardware.

Registration pattern used:

```python
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
...
register_cuda_ci(est_time=360, suite="stage-c-8-gpu-h100", labels=["short"])
register_rocm_ci(est_time=360, suite="stage-c-8-gpu-mi350", labels=["short"])
```

Suite mapping: the CUDA suite's GPU count is kept and the device tag is swapped
to `mi350` (`stage-c-8-gpu-h100` → `stage-c-8-gpu-mi350`; `stage-c-2-gpu-h200` →
`stage-c-2-gpu-mi350`; `stage-c-4-gpu-h200` → `stage-c-4-gpu-mi350`).

## Registered tests (verified GREEN on MI350X / gfx950)

| test | AMD suite | est_time / labels | evidence |
|------|-----------|-------------------|----------|
| `short/test_qwen2.5_0.5B_gsm8k_short` | `stage-c-8-gpu-mi350` | 360 / `[short]` | `Job` succeeded, step 2 `grad_norm` 3.34 |
| `short/test_qwen2.5_0.5B_gsm8k_async_short` | `stage-c-8-gpu-mi350` | 240 / `[short]` | `raysubmit_apNGPyKeDT3XKXMW` succeeded, steps 0/1/2 `grad_norm` 3.26 / 3.67 / 4.19 |
| `long/test_qwen2.5_0.5B_gsm8k_async` | `stage-c-2-gpu-mi350` | 5000 / `[long]` | `raysubmit_29gXFmNsfwbJTDBM` succeeded; re-confirmed with healthy steps 0–11 (`grad_norm` 1.03 / 1.09 / 0.86) |
| `short/test_run_megatron` | `stage-c-8-gpu-mi350` | 2000 / `[short]` | comparator `passed: true` (tp1 vs tp2pp2cp2, rel_diff ~3e-5), `[cli] Compare completed.`, exit 0 |
| `lora/test_lora_qwen2.5_0.5B` | `stage-c-4-gpu-mi350` | 300 / `[lora]` | `raysubmit_fpExE6d6PeHPTHcU` succeeded, steps 0–5 `grad_norm` 0.37 / 0.48 / 0.40, `eval/gsm8k` ~0.48; LoRA adapter load/unload to the sglang rollout engine works |

All runs were on the clean-build image `xinyujiangcmu/miles:rocm720-mi35x-20260615`
(8× MI350X), pass criterion = `Job '<id>' succeeded` **and** at least one
`train/step` with a finite `grad_norm`, no `Unable to find any suitable algorithms`,
no traceback at the end. The `test_run_megatron` comparator is judged on its own
exit-0 / all-passed contract instead.

## Honestly not registered (and why)

These were run or inspected and deliberately left out — recording the reasons so
the gap is explicit rather than silent.

- **`long/test_qwen2.5_0.5B_gsm8k` (sync) — FAIL, not registered.**
  Died at step 39/250 with a HIP OOM during the `--colocate` sleep/wake memory
  swap near the eval-interval-20 boundary (`Tried to allocate 5.14 GiB; GPU1 had
  6.00 GiB free, 5.25 GiB reserved-but-unallocated` → fragmentation).
  **This is not a fix/enablement regression:** steps 0–39 trained cleanly with
  finite `grad_norm`, and the same long+bridge family already passes GREEN as the
  **non-colocate** `test_qwen2.5_0.5B_gsm8k_async` (registered above) — async
  splits the 2 GPUs (1 actor / 1 rollout) while sync packs both onto the same two,
  so this is a 2-GPU colocate capacity/fragmentation limit, not a correctness
  problem. Mitigation is a separate change (see Follow-up), so the honest result
  here is an OOM fail.

- **`short/test_dumper` — SKIP.** Downloads the full `Qwen/Qwen3-30B-A3B`
  (~60 GB MoE) and runs a checkpoint conversion; out of scope for the
  small-model collection.

- **`megatron/test_qwen3_4B_ppo` — SKIP.** Already `disabled` on the CUDA side
  (`"PPO placement group has conflict on port, need fix later."`, with a
  `# FIXME`). Vendor-agnostic framework bug — there is no enabled CUDA baseline
  to mirror, so registering it on ROCm would be meaningless.

- **`megatron/test_qwen3_4B_p2p` — SKIP.** Already `disabled` on the CUDA side
  (`"RDMA weight update is not supported by current CI machine."`). The failure
  mode is missing RDMA on the runner, not an AMD-enablement signal.

- **`short/test_qwen3_0.6B_fsdp_colocated_2xGPU` and `tests/e2e/fsdp/*` — SKIP.**
  FSDP is not supported on AMD here (hangs at log_probs → Ray watchdog timeout).

## Follow-up (intentionally NOT in this PR)

1. **Wire the ROCm suites to runners.** `tests/ci/run_suite.py` `PER_COMMIT_SUITES`
   currently has `CPU` and `CUDA` keys but **no `HWBackend.ROCM` key**, so the
   registered `stage-c-*-mi350` suites currently hit the "Unknown suite" warning
   and are not executed by any job. Populating `PER_COMMIT_SUITES[ROCM]` and
   adding the matching `mi350` runner jobs in `.github/workflows/pr-test.yml` is
   the separate enablement step. The registrations in this PR are correct and
   recorded; they just don't *run* until the runner side lands.
2. **Confirm the `stage-c-4-gpu-mi350` tier.** The LoRA test introduces a 4-GPU
   mi350 suite (mirroring cuda `stage-c-4-gpu-h200`); confirm the exact name when
   the runner jobs are wired.
3. *(optional)* Revisit `long/test_qwen2.5_0.5B_gsm8k` (sync) for ROCm with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and/or a lower
   `--sglang-mem-fraction-static` (currently 0.7) to give the colocate train-phase
   wake_up enough headroom, then register if it reaches `Job succeeded`.

## Test plan

- Image: `xinyujiangcmu/miles:rocm720-mi35x-20260615` (clean from-Dockerfile build).
- Hardware: 8× MI350X (gfx950).
- Each candidate run one at a time; per-run logs kept under
  `test-runs/coll_<name>_20260615.log`.
- Only the registrations above (CUDA path untouched) are changed; the `mi350`
  suites are no-ops in CI until the runner wiring follow-up lands.
