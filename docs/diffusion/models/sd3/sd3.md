---
title: Stable Diffusion 3 / 3.5
description: SD3 family config, LoRA targets, and launch recipes for Flow-GRPO and DiffusionNFT.
---
## 1. Model introduction

[Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
is the primary SD3 checkpoint used in miles-diffusion recipes. SD3 uses a
DiT transformer with two text-conditioning inputs (`encoder_hidden_states` +
`pooled_projections`).

**Key highlights for RL training:**

- Single DiT component — weight sync targets `--update-weight-target-module transformer` (default).
- LoRA on the joint-attention `attn` projections of both streams (image `to_*`, text `add_*_proj` / `to_add_out`).
- Supports **Flow-GRPO** (OCR, HPS, or 0.8 OCR + 0.2 PickScore) and **DiffusionNFT** (PickScore) objectives.
- Gated Hugging Face model — requires `HF_TOKEN`.

## 2. Supported variants

| Model | HF ID | Notes |
|---|---|---|
| SD3.5 Medium | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | Default in all SD3 scripts |
| SD3 (other sizes) | Any checkpoint whose name contains `stable-diffusion-3` or `sd3` | Auto-detected by family resolver |

Override family detection when the checkpoint name does not match a registered
pattern (e.g. a renamed local directory):

```bash
--diffusion-model-family sd3
```

## 3. Environment setup

SD3.5 is gated — export your Hugging Face token:

```bash
export HF_TOKEN=<your_hf_token>
```

Optional prefetch (the scripts load the checkpoint by its Hub id, so it goes into the Hugging Face cache):

```bash
hf download stabilityai/stable-diffusion-3.5-medium
```

Prompt datasets live under
[`rockdu/miles-diffusion-datasets`](https://huggingface.co/datasets/rockdu/miles-diffusion-datasets):

| Recipe | Subset | Train path |
|---|---|---|
| GRPO + OCR | `flowgrpo_ocr` | `.../flowgrpo_ocr/train.jsonl` |
| GRPO + HPS | `hpdv2` | `.../hpdv2/train.jsonl` |
| NFT + PickScore | `flowgrpo_pickscore` | `.../flowgrpo_pickscore/train.jsonl` |

Launch scripts download the matching subset automatically via
`command_utils.hf_download_dataset`. Override the download root with
`--data-dir` (default `/root/datasets` →
`/root/datasets/miles-diffusion-datasets`).

## 4. Family config

Registered in `miles/backends/fsdp_utils/configs/sd3.py`:

```python
@register_train_pipeline_config("sd3")
class SD3TrainPipelineConfig(TrainPipelineConfig):
    hf_ckpt_name_patterns = ("stable-diffusion-3", "sd3")
    lora_target_modules = [
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    ]
```

Family resolution (`resolve_diffusion_model_family`) matches `--hf-checkpoint`
names against `hf_ckpt_name_patterns`. `--hf-checkpoint` is the single source
for the HF repo / local directory used by both FSDP training and
sglang-diffusion rollout; pass `--diffusion-model-family` only to override the
auto-detected family.

### SD3-specific training behavior

| Property | SD3 | Notes |
|---|---|---|
| Condition inputs | `encoder_hidden_states` + `pooled_projections` | Concatenated from rollout trajectory |
| CFG combine | `neg + scale * (pos - neg)` | Standard classifier-free guidance |
| Weight sync target | `transformer` (single component) | Multi-component models use comma-separated names |
| Rollout patch group | None | No extra sglang monkey patches |

Rollout responses serialize SD3 conditions as **single tensors** (not lists),
unlike some multi-encoder families.

## 5. Launch

All recipes are Python modules under `scripts/`. Each exposes a Typer CLI
(`ScriptArgs` dataclass) and submits training through
`command_utils.execute_train`. Common overrides: `--cuda-visible-devices`,
`--num-rollout`, `--data-dir`, `--extra-args`.

### 5.1 Script inventory

| Script | Reward | GPUs | Algorithm |
|---|---|---|---|
| `run_diffusion_grpo_sd3_ocr_sglang.py` | OCR (CPU) | 2 colocate | Flow-GRPO |
| `run_diffusion_grpo_sd3_hps_sglang.py` | HPS | 2 colocate | Flow-GRPO |
| `run_diffusion_grpo_sd3_ocr_pickscore_sglang.py` | 0.8 OCR + 0.2 PickScore | 2 colocate | Flow-GRPO |
| `run_diffusion_nft_sd3_pickscore.py` | PickScore | 3 (2+1) | DiffusionNFT |

### 5.2 Flow-GRPO + OCR (2 GPU colocate)

Canonical script: `scripts/run_diffusion_grpo_sd3_ocr_sglang.py`

**Status:** [🛡️ FG — Fully gated](/diffusion/user-guide/recipe-verification#fg)

```bash
export HF_TOKEN=...
python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py \
  --cuda-visible-devices 6,7
```

Walkthrough: [Quick Start](/diffusion/getting-started/quick-start).

E2E test: `tests/e2e/short/test_sd3_ocr_grpo_2xGPU.py`.


### 5.3 Flow-GRPO + HPS (2 GPU colocate)

Canonical script: `scripts/run_diffusion_grpo_sd3_hps_sglang.py`

**Status:** [📈 V — Verified](/diffusion/user-guide/recipe-verification#v) — 600 rollouts
(2 optimizer steps each, 1,200 in total) on 2×H200; `rollout/reward/raw_mean` 0.284 → 0.349
(mean of the last 100 rollouts, peak 10-rollout moving average 0.362).

```bash
export HF_TOKEN=...
python3 scripts/run_diffusion_grpo_sd3_hps_sglang.py \
  --cuda-visible-devices 6,7
```

The recipe uses the SD3 Flow-GRPO SDE, LoRA rank/alpha, IPC sync and precision settings of the OCR
recipe (not its `--sglang-lora-merge-mode merge`, `--rollout-fetch-in-parser`,
`--rollout-parser-num-workers 16`, or `--rollout-return-full-trajectory`), swaps in the deduplicated `hpdv2` prompts and `--rm-type hps`, and colocates one
HPS reward actor with the train and rollout workers. It keeps Flow-GRPO's own KL weight
(`--diffusion-kl-beta 0.01`) and group-wise advantage std instead of the OCR recipe's
`--diffusion-kl-beta 0.04 --globalize-reward-std`.

### 5.4 Flow-GRPO + OCR & PickScore (2 GPU colocate)

Script: `scripts/run_diffusion_grpo_sd3_ocr_pickscore_sglang.py`

**Status:** [📈 V — Verified](/diffusion/user-guide/recipe-verification#v) — 600 rollouts
(2 optimizer steps each) on 2×H200: `ocr_mean` 0.34 → 0.80, `pickscore_mean` 0.80 → 0.83,
`weighted_mean` 0.43 → 0.81 (means of the last 100 rollouts; 10-rollout moving-average peaks
0.88 / 0.85 / 0.87).

```bash
export HF_TOKEN=...
python3 scripts/run_diffusion_grpo_sd3_ocr_pickscore_sglang.py \
  --cuda-visible-devices 6,7
```

The OCR recipe with PickScore mixed in through `weighted_mixture_rm`
(`--custom-rm-args ocr=0.8,pickscore=0.2 --reward-key weighted`, the weighting Stepwise-Flow-GRPO
uses for its OCR task) and `--rollout-shuffle`; the PickScore actor is colocated next to the CPU
OCR pool. Every other flag matches the OCR recipe, except that it leaves `--sglang-lora-merge-mode` at
sgl-d's default `auto` instead of `merge`. See [Rewards](/diffusion/user-guide/rewards)
§ Combining rewards for how the mixture is wired.

### 5.5 DiffusionNFT + PickScore (3 GPU)

Script: `scripts/run_diffusion_nft_sd3_pickscore.py`

**Status:** [🛡️ FG — Fully gated](/diffusion/user-guide/recipe-verification#fg)

```bash
export HF_TOKEN=...
python3 scripts/run_diffusion_nft_sd3_pickscore.py \
  --cuda-visible-devices 4,5,2
```

Smoke test (1 rollout, OCR dataset, 2 GPUs):

```bash
MILES_SCRIPT_SMOKE=1 python3 scripts/run_diffusion_nft_sd3_pickscore.py
```

### Recipe comparison

| | GRPO + OCR | GRPO + HPS | GRPO + OCR & PickScore | NFT + PickScore |
|---|---|---|---|---|
| Script | `run_diffusion_grpo_sd3_ocr_sglang.py` | `run_diffusion_grpo_sd3_hps_sglang.py` | `run_diffusion_grpo_sd3_ocr_pickscore_sglang.py` | `run_diffusion_nft_sd3_pickscore.py` |
| `--loss-type` | `policy_loss` (default) | `policy_loss` (default) | `policy_loss` (default) | `nft` |
| SDE | Full window, noise=0.7, CFG=4.5 | Full window, noise=0.7, CFG=4.5 | Full window, noise=0.7, CFG=4.5 | ODE, noise=0 |
| Reference | LoRA base KL (β 0.04) | LoRA base KL (β 0.01) | LoRA base KL (β 0.04) | None; `π_old` is the EMA (`--use-ema`) |
| Reward placement | CPU OCR | Colocated HPS actor | CPU OCR + colocated PickScore actor | Dedicated PickScore GPU |
| Verification | FG | V | V | FG |

## 6. Recipe configuration

### GPU layout

**GRPO + OCR (default script):** 2 GPUs colocated (`--colocate`); OCR on CPU Ray actors.

**GRPO + HPS:** 2 GPUs colocated (`--colocate --hps-reward-colocate`); one HPS
actor shares the rollout GPUs.

**GRPO + OCR & PickScore:** 2 GPUs colocated (`--colocate --pickscore-reward-colocate`);
OCR on CPU Ray actors, one PickScore actor shares the rollout GPUs.

**NFT + PickScore:**

| GPU role | Count | Flags |
|---|---|---|
| FSDP train + sglang rollout (colocate) | 2 | `--actor-num-gpus-per-node 2`, `--rollout-num-gpus 2`, `--colocate` |
| PickScore reward | 1 | `--pickscore-num-workers 1`, `--pickscore-num-gpus-per-worker 1.0` |
| **Total** | **3** | `--num-gpus-per-node 3` |

PickScore runs as a Ray actor pool on a dedicated GPU. With `--pickscore-reward-colocate`
(not used in the default script), reward workers share the rollout GPUs
instead — useful only when GPU count is tight.

### Algorithm flags

**Flow-GRPO + OCR:**

| Setting | Value |
|---|---|
| Algorithm | `--loss-type policy_loss` (default) |
| Reference | `--diffusion-kl-beta 0.04` |
| Reward | `--rm-type ocr` |
| SDE | Full window (`num_sde_steps=10`, range `0,10`), noise 0.7, CFG 4.5 |
| Step strategy | `miles.rollout.step_strategy_hub.sde_window` |
| Weight sync | `--lora-ipc-weight-sync` (colocate IPC merge) |
| Determinism | `--deterministic-mode` (CI / e2e parity) |

**Flow-GRPO + HPS:** uses the same SDE, LoRA and precision flags as the OCR recipe,
with these settings of its own:

| Setting | Value |
|---|---|
| Reference | `--diffusion-kl-beta 0.01` |
| Advantage | Group-wise std (no `--globalize-reward-std`) |
| Reward | `--rm-type hps --hps-version v2.1` |
| Reward worker | One actor, batch size 8 |
| Placement | `--hps-reward-colocate` |
| Prompt subset | `hpdv2` |

**DiffusionNFT + PickScore:**

| Setting | Value |
|---|---|
| Algorithm | `--loss-type nft` |
| π_old | `--use-ema --rollout-weights ema` |
| Reward | `--rm-type pickscore` |
| SDE | `--diffusion-sde-type ode --diffusion-noise-level 0.0` |
| LoRA | rank 32, alpha 64, IPC sync |
| Precision | `--diffusion-forward-dtype fp16` |

## 7. LoRA and weight sync

All SD3 recipes use LoRA with IPC weight sync:

```bash
--use-lora \
--lora-ipc-weight-sync \
--lora-rank 32 \
--lora-alpha 64 \
--lora-init-weights gaussian \
--update-weight-buffer-size 2147483648
```

See [LoRA weight sync](/diffusion/advanced/lora) for how the weight-sync updater is selected
and the IPC merge internals.

## 8. Precision notes

All SD3 launch scripts use fp16 DiT forward:

```bash
--diffusion-forward-dtype fp16 \
--sglang-dit-precision fp16
```

SD3.5 fp16 policy gradients are small enough to underflow without scaling. When
`--diffusion-forward-dtype fp16`, the FSDP actor automatically enables
**`ShardedGradScaler`** around backward / optimizer step (no extra flag — see
`miles/backends/fsdp_utils/actor.py`). bf16/fp32 forward disables the scaler.

Flow-GRPO recipes also set **`--diffusion-clip-range`** (e.g. `1e-4` in the OCR
script) to clip importance ratios during the policy update.

Train/rollout dtype alignment for Flow-GRPO is covered in
[SDE step backend](/diffusion/advanced/sde-backend).

## 9. Reference results

### Flow-GRPO + OCR

Observed `rollout/reward/raw_mean` from one
`scripts/run_diffusion_grpo_sd3_ocr_sglang.py` run (default batch, 600 rollouts): the mean of the first 50 rollouts
is 0.462 and of the last 50 is 0.841. The curve is an example, not a CI acceptance range:

![Flow-GRPO OCR raw reward](/diffusion/assets/images/sd3/grpo-ocr-raw-reward.png)

### Flow-GRPO + OCR & PickScore

Observed rewards from one `scripts/run_diffusion_grpo_sd3_ocr_pickscore_sglang.py` run (600 rollouts);
the weighted sum GRPO trains on, then its two components:

![Flow-GRPO OCR + PickScore weighted reward](/diffusion/assets/images/sd3/grpo-ocr-pickscore-weighted-reward.png)

![Flow-GRPO OCR + PickScore: OCR component](/diffusion/assets/images/sd3/grpo-ocr-pickscore-ocr-reward.png)

![Flow-GRPO OCR + PickScore: PickScore component](/diffusion/assets/images/sd3/grpo-ocr-pickscore-pickscore-reward.png)

### DiffusionNFT + PickScore

Observed `rollout/reward/raw_mean` from one `scripts/run_diffusion_nft_sd3_pickscore.py` run (100 rollouts):

![DiffusionNFT PickScore raw reward](/diffusion/assets/images/sd3/nft-pickscore-raw-reward.png)

That run observed held-out
**`eval/pickscore_test` 0.850 / 0.859 / 0.863** at rollouts 29 / 59 / 89 with `--eval-interval 30` and 50 denoise steps
at eval time; these values are not asserted by the E2E fixture.

## 10. Pairs well with

- [Quick Start](/diffusion/getting-started/quick-start) — SD3.5 Flow-GRPO OCR walkthrough.
- [Rewards](/diffusion/user-guide/rewards) — OCR, HPS and PickScore scoring.
- [Customization](/diffusion/user-guide/customization) — `--*-path` plug-points.
- [SDE step backend](/diffusion/advanced/sde-backend) — SDE window (GRPO) vs ODE (NFT).
- [LoRA weight sync](/diffusion/advanced/lora) — IPC merge used by all SD3 recipes.
