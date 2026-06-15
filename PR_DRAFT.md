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

## Update — 2 enablement fixes + 3 more greens since first draft (13 GREEN total)

Beyond the pure-registration work, two small ROCm-gated, CUDA-byte-identical fixes were added
(each validated end-to-end on the hardware), turning previously-failing tests green:

1. **sglang test server FA3→triton on ROCm** (`tests/e2e/sglang/utils/sglang_server.py`):
   sglang's deterministic attention-backend auto-select has no HIP branch and defaults to `fa3`
   (NV-Hopper-only) on gfx9xx, crashing the server. Inject `--attention-backend triton` on ROCm only.
   → **`sglang/test_chat_input_ids_equivalence`** now PASS (registered).
2. **megatron Ray launch keeps HIP_VISIBLE_DEVICES for num_gpus=0 coordinators**
   (`miles/utils/external_utils/command_utils.py`): Ray blanks `HIP_VISIBLE_DEVICES=""` for the
   num_gpus=0 driver/coordinator process that Megatron `validate_args` probes, which is the
   `convert_checkpoint→train "No HIP GPUs"` blocker (investigate item #3 below — now fixed). Export
   `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` before `ray start --head`, ROCm-gated. Confirmed by an A/B
   probe + 4B phase-1 + v32/r3 reaching train. → **`megatron/test_mimo_7B_mtp_only_grad`** (7B) and
   **`megatron/test_qwen3_30B_A3B/test_baseline`** (30B-A3B MoE) now PASS (registered).

So the suite is now 13 registered GREEN (10 original + chat_input_ids + mimo_7B + 30B_A3B baseline).
The convert→train fix also clears the No-HIP-GPUs blocker for the rest of the qwen3_30B_A3B suite +
deepseek_v32, but those hit their own separate feature-specific gaps (routing-replay return-routed-experts,
deepep/fp8/int4 backends, sglang `DeepseekV32ForCausalLM` not registered, a TE `quantized_tensor` module
on the 4B-ckpt load path) — documented in RESULTS.md, not registered. See RESULTS.md for the full matrix.

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
- **`sglang/session_server/test_qwen3` — generation-backend 502 (platform gap, NOT flaky).** HTTP/CRUD layer OK
  (200/204) but every generate (`POST .../v1/chat/completions`) 502s; downstream "missing driver events ...
  events=[]" assertion accumulated >128k @ ~14-50/s over 45min (no model call ever completes). Server log "not
  supported on current platform" ×111 — suspected FA3-at-generate, same family as the chat_input_ids FA3 crash.
  `test_qwen35` / `test_glm47` deferred (same generation backend → same 502). Fix: default the sglang serving
  backend to `--attention-backend flash` on ROCm.
- **DeepSeek truncated CI tests — RUN, both FAIL (two distinct AMD gaps).** `deepseek_v4_flash_4layer_ci`: hard
  `ImportError: cannot import name 'fused_qk_rmsnorm' from aiter.ops.fused_qk_norm_rope_cache_quant` in sglang's
  `deepseek_common.attention_backend_handler` — the ROCm aiter build lacks the `fused_qk_rmsnorm` op DeepSeek-V4
  attention needs. `deepseek_v32_5layer_fp8`: gets past import, converts, then the train actor hits the same
  convert→train "No HIP GPUs" issue above. (These two were briefly mis-HELD as 744B-scale; they're ~28.6G
  truncated models — re-classified and run for real; disk reclaimed after.)
- **FSDP — unsupported on AMD (hangs).** `tests/e2e/fsdp/*` + `short/test_qwen3_0.6B_fsdp_colocated_2xGPU`.
- **CUDA-disabled (no enabled baseline to mirror; reasons are non-AMD):** `qwen3_4B_ppo` (PPO port bug),
  `qwen3_4B_p2p` (RDMA), `qwen3_5_35B_A3B_cp` (flaky), `quick_start_glm4_9B` (naive), `glm47_flash_ckpt` (bugs),
  `glm47_flash/test_r3_mtp_deepep` (deepep bug), `session_server/test_qwennext` (timeout),
  `qwen3_30B_A3B/test_r3_deepep_fp8` (fp8/bf16 mismatch), `precision/test_qwen3_0.6B_parallel_check` (bugs),
  `deepseek_v32_5layer_mxfp8` (superseded).
- **Held — genuinely too large for the shared disk (would endanger a co-tenant training job; need disk expansion
  / dedicated machine):** `glm5_744b_a40b_4layer*`, `kimi_k25_4layer`,
  `session_server/{minimax_m27 (~230B), nemotron3 (120B)}`. (The `deepseek_v4`/`deepseek_v32_5layer_fp8` tests were
  initially in this list by mistake — they're ~28.6G truncated CI models, since run for real, see above.)

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
