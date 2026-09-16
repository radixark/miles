---
title: Qwen-Image
description: Flow-GRPO with PickScore on Qwen-Image — the flow_grpo-aligned 5-GPU recipe.
---
## 1. Model introduction

[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) is Alibaba's MMDiT text-to-image model. In
miles-diffusion it is the **reference image recipe**: the launcher is aligned flag-for-flag with
flow_grpo's `pickscore_qwenimage` config, so a run here is directly comparable to the published
baseline.

**Key highlights for RL training:**

- **flow_grpo-aligned.** Per-prompt reward mean with a batch-wide std (`--globalize-reward-std`
  alone) is exactly flow_grpo's `PerPromptStatTracker` with `global_std=True`; `gaussian` LoRA
  init and `adam-beta2 0.999` match likewise.
- **RoPE caches rebuilt on CUDA.** diffusers builds `QwenEmbedRope`'s frequency tables on CPU
  while sglang-d builds them on CUDA; the fp32 ULP difference drifts every block. In one frozen-weight comparison,
  `noise_pred` mean |Δ| was approximately `2e-2`. `postprocess_model_after_materialize` rebuilds the caches on device
  after FSDP wrapping.
- **Variable-length text.** Conditioning is padded per batch with the attention mask derived
  from `txt_seq_lens` — the mask itself is never transmitted from rollout.

## 2. Supported variants

| Model | HF ID | Notes |
|---|---|---|
| Qwen-Image | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) | Family key `qwen_image` |

Any checkpoint whose name matches `qwen-image` resolves to the same config. For a
differently-named local directory — the usual case — add `--diffusion-model-family qwen_image`.

## 3. Family config

Registered in `miles/backends/fsdp_utils/configs/qwen_image.py`:

| Property | Value | Why |
|---|---|---|
| Timestep scaling | Trajectory timesteps ÷ 1000 before the DiT (`process_timestep_as_input`) | Matches how sglang-d's Qwen-Image pipeline rescales them |
| CFG combine | `uncond + scale·(cond − uncond)`, rescaled back to `‖cond‖` only when `true_cfg_scale > 1.0` | Mirrors sglang-d's `postprocess_cfg_noise`; the recipe's `true_cfg_scale 4.0` takes the rescaling branch |
| Cond collation | Pad `encoder_hidden_states` to batch max, mask from `txt_seq_lens`; honours `pad_to_len` | Legacy tiling path can reproduce whole-window padding bitwise |
| RoPE caches | Rebuilt on the model's CUDA device after materialize | CPU/CUDA `torch.pow` differ by fp32 ULPs |
| LoRA targets | `to_q to_k to_v to_out.0`, `add_q_proj add_k_proj add_v_proj to_add_out`, img/txt MLP proj | |
| Optimizer state | `transformer_blocks.59` text-branch params allowed missing | The last block's text outputs are discarded, so those params never receive gradient |

## 4. Launch

Canonical recipe: `scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py` — 4 colocate
GPUs + 1 PickScore GPU, 512×512, PickScore reward.

**Status:** [📈 V — Verified](/diffusion/user-guide/recipe-verification#v)

```bash
python3 scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py
```

Overrides go through the dataclass CLI, e.g.
`--num-rollout 50 --extra-args "--diffusion-kl-beta 0.02"`.

## 5. Recipe notes

- Of the 10 denoising steps, only steps 3-4 are trained: `--diffusion-sde-window-range 3,5` with
  `--diffusion-num-sde-steps 2` leaves the window nowhere to move, so every sample trains the
  same two mid-schedule steps as SDE while the rest run ODE.
- Batch arithmetic: 32 prompts × 16 samples = 512 samples per rollout, split into two optimizer
  steps of 256 samples; the 2 trained SDE steps expand those to 1024 train pairs.
- `--rollout-patch-group sgld` applies the diffusers-op-parity patches on the engine. It costs a
  little rollout throughput and buys a much smaller `train/log_prob_mean_abs_diff`.

## 6. Pairs well with

- [LoRA weight sync](/diffusion/advanced/lora) — `--lora-ipc-weight-sync` is on in this recipe.
- [Dtype Control](/diffusion/advanced/dtype-control) — why fp32 master + bf16 forward.
- [Deterministic Training](/diffusion/advanced/deterministic) — `--deterministic-mode` is on.
