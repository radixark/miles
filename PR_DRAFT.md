# [AMD] Seed the ROCm e2e CI suite with verified MI350X tests

## What this does

Registers the miles e2e tests that pass end-to-end on AMD MI350X (gfx950) into the
`HWBackend.ROCM` CI suite, by adding a `register_rocm_ci(...)` marker next to each
test's existing `register_cuda_ci(...)`. Each ROCm registration mirrors the CUDA one
(same `est_time`, same `labels`) and points at the matching `stage-c-*-mi350` suite.
The `register_rocm_ci` + `HWBackend.ROCM` framework lives in `tests/ci/ci_register.py`
(parsed by the same AST machinery as `register_cuda_ci`). Nothing is registered that
hasn't actually run green on the hardware.

Registration pattern:

```python
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
...
register_cuda_ci(est_time=360, suite="stage-c-8-gpu-h100", labels=["short"])
register_rocm_ci(est_time=360, suite="stage-c-8-gpu-mi350", labels=["short"])
```

Suite mapping keeps the CUDA suite's GPU count and swaps the device tag to `mi350`
(`-h100`/`-h200` → `-mi350`).

## Registered tests (verified GREEN on MI350X / gfx950)

| test | AMD suite | est_time / labels |
|------|-----------|-------------------|
| `short/test_qwen2.5_0.5B_gsm8k_short` | `stage-c-8-gpu-mi350` | 360 / `[short]` |
| `short/test_qwen2.5_0.5B_gsm8k_async_short` | `stage-c-8-gpu-mi350` | 240 / `[short]` |
| `long/test_qwen2.5_0.5B_gsm8k_async` | `stage-c-2-gpu-mi350` | 5000 / `[long]` |
| `short/test_run_megatron` | `stage-c-8-gpu-mi350` | 2000 / `[short]` |
| `lora/test_lora_qwen2.5_0.5B` | `stage-c-4-gpu-mi350` | 300 / `[lora]` |
| `sglang_config/test_sglang_config` | `stage-c-8-gpu-mi350` | 600 / `[short]` |
| `sglang_config/test_sglang_config_mixed_offload` | `stage-c-8-gpu-mi350` | 300 / `[short]` |
| `sglang_config/test_sglang_config_mixed_offload_ft` | `stage-c-8-gpu-mi350` | 600 / `[short]` |
| `precision/test_hf_attention_cp_relayout` | `stage-c-4-gpu-mi350` | 60 / `[precision]` |
| `precision/test_qwen3_5_cp_correctness` | `stage-c-4-gpu-mi350` | 120 / `[precision]` |

Pass criterion for training tests = `Job '<id>' succeeded` and a `train/step` with finite
`grad_norm`, no `Unable to find any suitable algorithms`, no end traceback. The comparator
and precision tests are judged on their own exit-0 / all-passed contracts. Evidence (job ids,
grad_norms, comparator diffs) is in the collection log `amd-ci-worker/RESULTS.md`.

## Honestly not registered (with reasons)

- **Megatron training with `convert_checkpoint` — `RuntimeError: No HIP GPUs are available` (genuine).**
  Every megatron e2e that runs `U.convert_checkpoint` (HF → torch_dist) in `prepare()` fails: the convert
  step completes fine (builds the model on N ranks), but the subsequent `execute_train` Ray actor dies at
  Megatron `validate_args` → `get_device_capability/get_device_properties` → "No HIP GPUs available", 0 steps.
  Affected: `ckpt/test_qwen3_4B_ckpt` (4B), `megatron/test_mimo_7B_mtp_only_grad` (7B), the whole
  `megatron/test_qwen3_30B_A3B/*` suite + `short/test_dumper` (30B). The GREEN megatron tests above (all 0.5B)
  use `--ref-load <HF dir>` directly (no convert) and their train actor gets GPUs fine. **The trigger is
  convert_checkpoint, not model size** — likely the convert step leaves GPU/Ray placement state such that the
  post-convert train actor is assigned 0 GPUs on AMD. Environment was verified healthy throughout (0.5B megatron
  re-passes from a clean Ray state; a `@ray.remote(num_gpus=1)` probe sees a GPU). **High-value single fix** —
  would unblock 4B + 7B + the entire 30B-MoE suite at once.
- **`long/test_qwen2.5_0.5B_gsm8k` (sync) — colocate OOM (env, not a fix bug).** Died at step 39/250 with a HIP
  OOM during the `--colocate` sleep/wake swap; the non-colocate `_async` sibling (registered) proves the
  long+bridge path works on AMD. Mitigation (`expandable_segments` / lower `--sglang-mem-fraction-static`) is a
  separate change.
- **`sglang/test_chat_input_ids_equivalence` — sglang-serving FA3.** The `start_sglang_server` test util launches
  `sglang.launch_server` without `--attention-backend`, so it defaults to FA3 (FlashAttention-3, NV-Hopper-only)
  → `ImportError: Can not import FA3 in sgl_kernel`. The util should pass `--attention-backend flash` (which the
  training tests' rollout engine already uses successfully on AMD).
- **`sglang/session_server/*` — inconclusive / slow.** `test_qwen3` (30B-FP8) is not FA3-blocked (uses the miles
  rollout engine, GPUs busy throughout) but its session-verify ran 47 min without completing (est 400s; >7x over
  budget — possible AMD FP8-serving perf gap). `test_qwen35` (35B-FP8) / `test_glm47` deferred as the same slow
  path + large downloads.
- **FSDP — unsupported on AMD (hangs).** `tests/e2e/fsdp/*` + `short/test_qwen3_0.6B_fsdp_colocated_2xGPU`.
- **CUDA-disabled (no enabled baseline to mirror; reasons are non-AMD):** `qwen3_4B_ppo` (PPO port bug),
  `qwen3_4B_p2p` (RDMA), `qwen3_5_35B_A3B_cp` (flaky), `quick_start_glm4_9B` (naive), `glm47_flash_ckpt` (bugs),
  `glm47_flash/test_r3_mtp_deepep` (deepep bug), `session_server/test_qwennext` (timeout),
  `qwen3_30B_A3B/test_r3_deepep_fp8` (fp8/bf16 mismatch), `precision/test_qwen3_0.6B_parallel_check` (bugs),
  `deepseek_v32_5layer_mxfp8` (superseded).
- **Held — exceed shared-disk headroom (would endanger a co-tenant training job; need disk expansion / dedicated
  machine):** `glm5_744b_a40b_4layer*`, `kimi_k25_4layer`, `deepseek_v32_5layer_{fp8,mxfp8}`,
  `deepseek_v4_flash_4layer`, `session_server/{minimax_m27 (~230B), nemotron3 (120B)}`.

## Follow-up (not in this PR)

1. **Wire the ROCm suites to runners.** `tests/ci/run_suite.py` `PER_COMMIT_SUITES` has no `HWBackend.ROCM` key,
   so the registered `stage-c-*-mi350` suites currently hit the "Unknown suite" warning and aren't executed by
   any job. Populate `PER_COMMIT_SUITES[ROCM]` and add the matching `mi350` runner jobs in
   `.github/workflows/pr-test.yml`. (Registrations here are correct/recorded; they just don't *run* until then.)
2. **Confirm the `stage-c-4-gpu-mi350` tier name** (introduced by the LoRA + precision tests, mirroring
   cuda `stage-c-4-gpu-h200`).
3. **Investigate the convert_checkpoint → train "No HIP GPUs" issue** (placement / `HIP_VISIBLE_DEVICES` of the
   post-convert train Ray actor) — single highest-value AMD enablement fix here.
4. *(optional)* `start_sglang_server` util: default `--attention-backend flash` on ROCm so the sglang-serving
   e2e tests run; long-sync colocate mem mitigation.

## Test plan

- Image: `xinyujiangcmu/miles:rocm720-mi35x-20260615` (clean from-Dockerfile build); 8× MI350X (gfx950).
- One test at a time, GPU-courtesy-checked, with a thorough Ray cleanup before each (the container has no
  init-reaper, so Ray clusters/zombies accumulate; `ray stop --force` ×3 + kill keeps GPU placement sane).
- Per-run logs under `test-runs/coll_*_20260615.log`; full pass/fail/skip ledger in `amd-ci-worker/RESULTS.md`.
- Only the registrations above change (CUDA path untouched); the `mi350` suites are no-ops in CI until the
  runner-wiring follow-up lands.
