---
title: LTX-2
description: Video GRPO on LTX-2.3 — native model package, CPS-SDE, unguided forward.
---
## 1. Model introduction

[LTX-2](https://github.com/Lightricks/LTX-2) is Lightricks' audio-video DiT. LTX-2.3 is the
variant miles-diffusion trains, and the framework's most unusual family: the only one that does
**not** go through diffusers, the only one trained **unguided**, and the only one on the **CPS**
SDE kernel.

**Key highlights for RL training:**

- **Native model package, not diffusers.** `miles/backends/fsdp_utils/models/ltx/` supplies
  loading, modeling, attention, and the FSDP plan directly against `ltx_core`; FSDP wraps
  `BasicAVTransformerBlock`. Requires `ltx-core` (pinned in `requirements.txt`, baked into the
  Docker image).
- **Unguided training.** `supports_cfg_training = False`: any `--diffusion-guidance-scale` other
  than `1.0`, or a `--diffusion-negative-prompt`, is a parse-time error. The training forward is
  a single velocity pass — there is no negative branch to combine.
- **Video branch only.** The audio stream is loaded but never trained
  (`optimizer_state_allowed_missing = ["audio"]`), and `--rollout-patch-group ltx` disables
  audio-video cross-attention on the engine to match.
- **No sequence parallelism.** `sequence_parallel_plan` raises `NotImplementedError`; leave
  `--sequence-parallel-size` at `1`.

## 2. Supported variants

| Model | HF ID | Notes |
|---|---|---|
| LTX-2.3 | [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) | Video branch only |

## 3. Family config

Registered in `miles/backends/fsdp_utils/configs/ltx.py`:

| Property | Value | Why |
|---|---|---|
| Velocity → noise | `forward_velocity` reconstructs x0 via `ltx_core.utils.to_denoised` in fp32 and divides back out | Algebraically an identity, but the fp32 rounding path is what the e2e standards were recorded against |
| Boundary dtypes | `latents` / `cond` cast to the forward dtype, `timestep` passthrough | Element-wise math anchors on `latents.dtype` and rollout runs bf16 — see [Dtype Control](/diffusion/advanced/dtype-control) |
| Precision | bf16 end to end (master, reduce, forward, engine) + math-SDPA on both sides | Matches the original bitwise E2E reference; math-SDPA is deterministic by construction |
| SDE | CPS kernel, `sde_timestep_divisor = 1000.0` | σ comes straight from the carried rollout timesteps rather than a scheduler lookup; log-prob drops its constants |
| LoRA targets | Attention + FFN: `to_q`, `to_k`, `to_v`, `to_out.0`, `net.0.proj`, `net.2` | |
| Rollout patches | `--rollout-patch-group ltx`: `patch_ltx2_rollout_cond_kwargs`, `patch_ltx2_disable_av_cross` | Video-only training forward needs AV cross-attention off engine-side |

## 4. Launch

Canonical recipe: `scripts/run_diffusion_grpo_ltx23_sglang.py` — a 5-GPU Ray node with 4 GPUs shared by FSDP and
rollout plus 1 separately scheduled PickScore GPU, 57 frames @ 24 fps, 512×768.

**Status:** [🛡️ FG — Fully gated](/diffusion/user-guide/recipe-verification#fg)

```bash
python3 scripts/run_diffusion_grpo_ltx23_sglang.py
```

## 5. Recipe notes

- The step strategy is `epoch_global_random_choice`: 3 SDE steps drawn once per epoch from
  candidates 0-9 and shared by every sample — with only 64 samples per rollout, per-request
  randomization would add variance the batch is too small to average out.
- The clip range is `1e-5`, an order of magnitude tighter than the image recipes' `1e-4`.
- `--pickscore-num-frames 3` scores three evenly-spaced frames per video; scoring all 57 would
  dominate rollout time for a reward signal that barely changes.
- Health checks are relaxed (`--rollout-health-check-interval 120`, router failure threshold 30):
  a 57-frame request takes minutes, and the image-recipe defaults would kill healthy engines
  mid-generation.
- `--rollout-parser-num-workers 8`: 57-frame trajectory tensors are large enough that one
  deserializer actor starves the engines.

## 6. Reference results

The original bf16-master 4-train-GPU + 1-reward-GPU run, used as the first LTX E2E
reference, raises `rollout/reward/raw_mean` from ~0.68 to ~0.77 over 200 rollouts:

![LTX-2.3 PickScore reward mean](/diffusion/assets/images/ltx/reward_mean.png)

A future LTX update will make `train/model_output_mean_abs_diff` **0.0** and switch
`--fsdp-master-dtype` to `fp32`.

## 7. Pairs well with

- [Dtype Control](/diffusion/advanced/dtype-control) — the boundary-dtype policy is why LTX has one.
- [Deterministic Training](/diffusion/advanced/deterministic) — `sdpa_math` is the deterministic-safe
  backend this recipe relies on.
