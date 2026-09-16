---
title: Wan2.2-T2V-A14B
description: Dual-expert MoE video model — Flow-GRPO + PickScore recipe, LoRA SFT recipe, and the high/low-noise expert boundary.
---
## 1. Model introduction

[Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) is a text-to-video model with a **dual-expert
MoE DiT**: a high-noise expert (`transformer`) denoises timesteps `t ≥ boundary` and a low-noise expert
(`transformer_2`) handles the rest. Conditioning comes from a UMT5 text encoder; latents go through the Wan VAE (4×
temporal compression).


**Key highlights for RL training:**

- **Two experts, one boundary.** `boundary_ratio = 0.875` — which expert a train pair updates depends only on its
  timestep. Single- and dual-expert training are both supported: `--update-weight-target-module` names the experts to
  load, train, and sync.
- **Two guidance scales.** Rollout denoises low-noise steps with `guidance_scale_2` and there is **no fallback** —
  training asserts `--diffusion-guidance-scale-2` is set explicitly, because a silent mismatch against rollout would
  corrupt the ratio.
- **USP-ready.** Wan was enabled for Ulysses × Ring sequence parallelism.

## 2. Supported variants

| Model | HF ID | Notes |
|---|---|---|
| Wan2.2-T2V-A14B | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Default in both Wan recipes |


## 3. Family config

Registered in `miles/backends/fsdp_utils/configs/wan2_2.py`:

| Property | Value | Notes |
|---|---|---|
| Expert routing | `t ≥ 0.875 × num_train_timesteps` → `transformer`, else `transformer_2` | `component_for_timestep` |
| Guidance routing | high-noise → `--diffusion-guidance-scale`, low-noise → `--diffusion-guidance-scale-2` (required) | `select_guidance_scale` |
| Timestep scaling | None — Wan DiT takes raw scheduler timesteps (0..1000) | |
| Condition inputs | `encoder_hidden_states` only (fixed-length UMT5 embeds) | Concat-collate, no padding needed |
| CFG combine | `neg + scale × (pos − neg)` | Standard |
| CFG batching | Off | |
| FSDP fp32 param map | `norm2` parameters only | `models/diffusers/wan2_2/parallel_plan.py`; the separate rollout patch group aligns selected norm and `scale_shift_table` arithmetic engine-side |

## 4. Launch
### 4.1 Flow-GRPO + PickScore (4 train GPUs + 1 reward GPU)

Canonical recipe: `scripts/run_diffusion_grpo_wan22_pickscore_5gpu.py`

**Status:** [○ NV — Not verified](/diffusion/user-guide/recipe-verification#nv)

```bash
python3 scripts/run_diffusion_grpo_wan22_pickscore_5gpu.py
```


### 4.2 Full-finetune Flow-GRPO + PickScore, multi-node (2×8 train GPUs + 1 reward GPU)

Recipe: `scripts/run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py`
(full finetune, no LoRA, true on-policy). Start the
[multi-node Ray cluster](/diffusion/user-guide/launch-script#multi-node-training), then run on the head node:

**Status:** [🧩 PG — Proxy gated](/diffusion/user-guide/recipe-verification#pg)

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py
```

`--four-gpu-ci` scales the recipe to one 4-GPU node for e2e CI.

### 4.3 LoRA SFT on (video, prompt) pairs (4 GPUs, no rollout engines)

Recipe: `scripts/run_diffusion_sft_wan22.py`

**Status:** [○ NV — Not verified](/diffusion/user-guide/recipe-verification#nv)

```bash
MILES_SCRIPT_DATA_JSONL=/abs/data.jsonl python3 scripts/run_diffusion_sft_wan22.py
```

## 5. Reference results

### 5.1 17-GPU multi-node full-finetune GRPO

#### CFG enabled (`4.0/3.0`)

- `train/model_output_mean_abs_diff`: **0.0** every rollout.
- `rollout/reward/raw_mean`: ~0.77 → ~0.82.
- Final `eval/pickscore_test`: **0.8231** (rollout 199, 28 denoising steps—not 40,
  UniPC, 2048 prompts).

![Wan2.2 CFG PickScore reward mean](/diffusion/assets/images/wan/reward_mean.png)

#### CFG disabled (`1.0/1.0`)

- `rollout/reward/raw_mean`: ~0.72 → ~0.83.
- Final `eval/pickscore_test`: **0.8243** (rollout 199, 28 denoising steps—not 40,
  UniPC, 2048 prompts).

![Wan2.2 no-CFG PickScore reward mean](/diffusion/assets/images/wan/reward_mean_nocfg.png)

## 6. Pairs well with

- [Single-Prompt Multi-Generation](/diffusion/advanced/single-prompt-multi-gen) — the microgroup mechanics behind
  `--rollout-microgroup-size 8`.
- [LoRA Training and Weight Sync](/diffusion/advanced/lora) — IPC merge used by the GRPO recipe.
- [SDE Step Backend](/diffusion/advanced/sde-backend) — how the trained SDE step is scored train-side.
- [Rewards](/diffusion/user-guide/rewards) — PickScore worker pool configuration.
