---
title: FLUX 3 Image Klein
description: FLUX 3 Image Klein DiffusionNFT with PickScore (LoRA) and full-finetune Flow-GRPO with HPSv2.1, both fully colocated on 8 GPUs.
---

## 1. Model introduction

FLUX 3 Image Klein is a flow-matching DiT for text-to-image generation and image editing: per-stream mode blocks for the
image and text tokens, then joint single-stream blocks over `[image | text]`, with 4-axis RoPE and per-token AdaLN
modulation. Miles supports RL on the text-to-image task only (`--diffusion-task t2i`); image-conditioned requests are
not supported. Rollouts run on sglang-diffusion's native FLUX 3 Image Klein pipeline (`Flux3ImagePipeline`) at
1024×1024 in bf16; the trainer runs the `MilesModelBackend` package `miles/backends/fsdp_utils/models/flux3_image/`,
which builds only the text-to-image path.

## 2. Supported variants

| Variant | Checkpoint | Status |
| --- | --- | --- |
| Klein Base (CFG, not step-distilled) | [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) (gated: needs `HF_TOKEN`) | Recipes below |

The step-distilled Klein variant samples without CFG through a different denoise loop and is not supported.
The launchers' `--model-path` option also takes a local package directory.

## 3. Family config

From `miles/backends/fsdp_utils/configs/flux3_image.py`:

| Property | Value | Why |
| --- | --- | --- |
| Rollout microgroup | 1 (validated) | The engine's rollout path accepts one batch with one output per request |
| Micro-batch | 1 (validated) | Text length varies per prompt and the DiT takes no padding mask |
| Attention | `--fsdp-attention-backend _flash_3` on Hopper (SM9x), `_flash_4` on SM10x Blackwell; the trainer rejects other GPUs | The engine's default flash kernel on each GPU; its backward is deterministic under `--deterministic-mode` (the Flow-GRPO recipe sets it, the DiffusionNFT recipe does not) |
| CFG off | `--diffusion-guidance-scale 1` requires `--rollout-patch-group flux3_image` | Unpatched, the engine runs the whole CFG pair even at a scale of 1, packed into one forward over both branches: not bitwise the conditional velocity |
| CFG combine | `neg + scale · (pos − neg)`, two batch-1 forwards; `true_cfg_scale` wins when set | Same formula as the engine |
| Full-weight sync | Weight bridge renames `*.{query,key}_norm.scale` to `.weight` | The engine's tensor-update path skips names its module lacks |

The `flux3_image` rollout patch group (`miles/backends/sglang_diffusion_utils/monkey_patches/patch_flux3_image_cfg_off.py`)
runs a conditional-only denoise loop when the CFG scale is 1 or less: one DiT forward per step, the rollout transition
hook intact, CFG-on requests unchanged.

## 4. Launch

Both recipes put training, rollout and the reward model on the same 8 GPUs (`--colocate` plus
`--<reward>-reward-colocate`, one reward worker per engine) and decode rollout responses in parallel parser actors
(`--sglang-server-concurrency 4 --rollout-fetch-in-parser --rollout-parser-num-workers 32`). The recipes target
Hopper; on SM10x Blackwell, append `--extra-args "--fsdp-attention-backend _flash_4"`.

### 4.1 DiffusionNFT + PickScore, LoRA, 8 GPUs colocated

`scripts/run_diffusion_nft_flux3_image_pickscore_8gpu.py` — LoRA rank 32 / alpha 64 synced over CUDA IPC in
`dynamic` merge mode (the engine keeps the adapter unmerged, like the trainer), an EMA rollout policy and deterministic
ODE rollouts at 16 steps without CFG: NFT trains the unguided velocity, so the rollout samples it too. 8 prompts × 8
samples per rollout, lr 3e-4, gradient checkpointing.

**Status:** [📈 V — Verified](/diffusion/user-guide/recipe-verification#v)

```bash
python3 scripts/run_diffusion_nft_flux3_image_pickscore_8gpu.py
```

### 4.2 Flow-GRPO + HPSv2.1, full finetune, 8 GPUs colocated

`scripts/run_diffusion_grpo_flux3_image_hps_8gpu.py` — full finetune (no LoRA) with fp32 master and reduce, bf16
forward, lr 1e-5; full weights sync over CUDA IPC. 10 sampling steps with CFG 3.5 on both the rollout and the trainer,
Flow-SDE noise level 0.7 on one SDE step per rollout drawn from steps 1-2, clip 1e-4, no KL. Old log-probs come from
the rollout: with this noise level the train-rollout drift leaves the ratio at 1 (`approx_kl` at most ~1e-12, `clipfrac` 0).
16 prompts × 16 samples per rollout, gradient checkpointing.

**Status:** [📈 V — Verified](/diffusion/user-guide/recipe-verification#v)

```bash
python3 scripts/run_diffusion_grpo_flux3_image_hps_8gpu.py
```

## 5. Reference results

Both runs use the default seed and the recipe defaults on one 8×H200 node, except that the Flow-GRPO run stops at
150 rollouts (`--num-rollout 150`; the recipe default is 200). Held-out numbers use the first 32 prompts
of each reward's test split, one image per prompt at a fixed seed, at the training step count.

### 5.1 DiffusionNFT + PickScore

The PickScore reward (`rollout/reward/raw_mean`, PickScore / 26) rises from 0.769 (mean of the first 10 rollouts) to
0.857 (mean of the last 10) over 100 rollouts:

![FLUX 3 Image Klein DiffusionNFT PickScore reward mean](/diffusion/assets/images/flux3-image/nft_pickscore_reward_mean.png)

On held-out PickScore prompts at 16 steps, the training setting. Klein Base's own default is 50 steps with CFG 3.5,
so neither base row shows its default quality:

| Model | Sampling | PickScore / 26 |
| --- | --- | --- |
| Base | CFG 3.5 | 0.833 |
| Base | CFG off | 0.775 |
| NFT, rollout 100 | CFG off | **0.878** |

The trained model without CFG beats the base model without CFG on 32/32 prompts and the base model with CFG 3.5 on
27/32. It also drifts toward PickScore's preferences: brighter, more saturated images, and on some prompts it adds or
beautifies people.

Case studies (base with CFG 3.5, base without CFG, NFT without CFG):

*An impressionist portrait of a shar pei*

![shar pei: base CFG 3.5, base CFG off, NFT](/diffusion/assets/images/flux3-image/nft_shar_pei.webp)

*A digital illustration of a dragon flying over a castle, fantasy style*

![dragon over a castle: base CFG 3.5, base CFG off, NFT](/diffusion/assets/images/flux3-image/nft_dragon_castle.webp)

*an epic angel dressed in blue with white wings*

![angel: base CFG 3.5, base CFG off, NFT](/diffusion/assets/images/flux3-image/nft_angel.webp)

*enigmatic black square building on top of a purple hill, smoke stacks*

![black building on a purple hill: base CFG 3.5, base CFG off, NFT](/diffusion/assets/images/flux3-image/nft_black_building.webp)

### 5.2 Flow-GRPO + HPSv2.1, full finetune

The HPSv2.1 reward rises from 0.263 (mean of the first 10 rollouts) to 0.278 (mean of the last 10) over 150 rollouts;
it is flat for the first ~20 rollouts, as expected of a full finetune at lr 1e-5 with clip 1e-4:

![FLUX 3 Image Klein Flow-GRPO HPSv2.1 reward mean](/diffusion/assets/images/flux3-image/grpo_hps_reward_mean.png)

On held-out HPDv2 prompts at 10 steps with CFG 3.5 (the training settings, not Klein Base's 50-step default),
HPSv2.1 goes from 0.270 (base) to 0.282 (rollout 150), higher on 29/32 prompts. The gain is small but consistent:
images keep their composition and gain detail and color.

Case studies (base and Flow-GRPO, both with CFG 3.5):

*Anthropomorphic alien plant creature with big eyes and leafy limbs depicted in a detailed painting.*

![alien plant creature: base, Flow-GRPO](/diffusion/assets/images/flux3-image/grpo_alien_plant.webp)

*A massive and brightly colored spacecraft in a deserted landscape, depicted in retro 1960s sci-fi art.*

![retro spacecraft: base, Flow-GRPO](/diffusion/assets/images/flux3-image/grpo_retro_spacecraft.webp)

*A cyber girl with demon horns holds a black feather in front of a cybercity with a gloomy expression.*

![cyber girl: base, Flow-GRPO](/diffusion/assets/images/flux3-image/grpo_cyber_girl.webp)

## 6. Pairs well with

- [LoRA Training and Weight Sync](/diffusion/advanced/lora) — IPC adapter sync and merge modes.
- [Rewards](/diffusion/user-guide/rewards) — PickScore and HPSv2.1.
