---
title: Quick Start
description: A working Flow-GRPO training job on SD3.5 + OCR — default script uses 2 GPUs, using the Flow-GRPO OCR dataset (`flowgrpo_ocr`).
---
This page takes you from `docker pull` to a running **Flow-GRPO** job on
**Stable Diffusion 3.5 Medium** with **OCR** reward. The launch script below
targets **2 GPUs** (colocate train + rollout). A single-GPU topology is not
provided or verified. You need Hugging Face access to the gated SD3.5 checkpoint.

Installation and environment setup are documented separately — this page starts
inside a ready container or machine.

Deeper SD3 recipe config, reference curves, and the DiffusionNFT + PickScore
recipe are in the [SD3 model guide](/diffusion/models/sd3/sd3).

## 1. Start the container

On the **host** (experimental image — see
[`docker/README.md`](https://github.com/radixark/miles_diffusion/blob/main/docker/README.md)):

```bash
docker pull radixark/miles_diffusion:latest

docker run --rm \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -it radixark/miles_diffusion:latest /bin/bash
```

Refresh the editable install inside the container:

```bash
cd /root/miles_diffusion && git pull && pip install -e . --no-deps
```

Steps 2–4 below run inside the container (or on any machine with the same deps).

<Note>
Launch recipes live under `scripts/` as **Python modules** (not bash). Each
script sets recipe-specific env vars (e.g. `PYTORCH_CUDA_ALLOC_CONF`) and submits `train_diffusion.py` through Ray via
`miles.utils.external_utils.command_utils`.
</Note>

## 2. Download model and data

SD3.5 is a **gated** Hugging Face model. Export your token before any download or
training:

```bash
export HF_TOKEN=<your_hf_token>
```

The launch script loads the checkpoint by its Hub id and downloads it into the Hugging Face cache on
first run. To prefetch into that cache:

```bash
hf download stabilityai/stable-diffusion-3.5-medium
```

Training prompts come from the **`flowgrpo_ocr`** subset of
[`rockdu/miles-diffusion-datasets`](https://huggingface.co/datasets/rockdu/miles-diffusion-datasets).
Each prompt embeds the target string in double quotes (e.g. a logo saying
`"Miles"`) — OCR reward compares PaddleOCR output against that target.

The script downloads the subset under `--data-dir` (default `/root/datasets`) on every launch;
`hf download` skips files that are already there. To prefetch into the default location:

```bash
hf download --repo-type dataset rockdu/miles-diffusion-datasets \
  --include "flowgrpo_ocr/**" \
  --local-dir /root/datasets/miles-diffusion-datasets
```

| Split | Path |
|---|---|
| Train | `<data-dir>/miles-diffusion-datasets/flowgrpo_ocr/train.jsonl` |

See [Rewards](/diffusion/user-guide/rewards) for OCR scoring and prompt format.

## 3. Launch training

```bash
export HF_TOKEN=<your_hf_token>
python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py \
  --cuda-visible-devices 6,7
```

The script starts Ray, launches sglang-diffusion rollout engines, loads the
FSDP actor with LoRA, syncs weights via CUDA IPC (`--lora-ipc-weight-sync`),
and runs the Flow-GRPO rollout / train loop. With `WANDB_API_KEY` set, images
and metrics are logged to project `miles-diffusion-grpo`.

Once rollout begins, stdout prints a `[reward stats] raw mean=…` line per rollout and
`[train step N]` lines with metrics such as `train/loss` and `train/log_prob_mean_abs_diff`.
WandB (when configured) also gets the reward as `rollout/reward/raw_mean`.

To finish faster while debugging, override rollout count (default **600**):

```bash
python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py \
  --cuda-visible-devices 6,7 \
  --num-rollout 2
```

Any `ScriptArgs` field also accepts `MILES_SCRIPT_<FIELD>` (e.g.
`MILES_SCRIPT_NUM_ROLLOUT=2`). For train/rollout alignment debugging (skips the
optimizer step; does not run IPC weight-sync checksums):

```bash
MILES_SCRIPT_DEBUG_ALIGNMENT=1 python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py
```

## 4. What's happening

Each rollout iteration runs four steps:

```mermaid
flowchart LR
    P[flowgrpo_ocr] --> R[sglang-diffusion rollout]
    R --> RM[OCR rm_hub]
    RM --> A[FSDP actor + LoRA]
    A == CUDA IPC weight sync ==> R
    A -. KL .-> Ref[(LoRA base)]
```

1. Sample prompts and generate images via sglang-diffusion (SDE window stepping).
2. Score images with OCR (`--rm-type ocr`) on CPU Ray actors — no reward GPU.
3. Compute the **Flow-GRPO** objective (GRPO advantage normalization + SDE
   log-prob ratios) and step the LoRA optimizer.
4. Push updated LoRA weights to rollout engines via `--lora-ipc-weight-sync`.

This recipe uses **Flow-GRPO** (`--loss-type policy_loss`, the default) with
LoRA-base KL (`--diffusion-kl-beta 0.04`), noise level 0.7, CFG 4.5, and batch-wide reward-standard-deviation
normalization (`--globalize-reward-std`). `--deterministic-mode` and `--global-batch-size 64` match the CI e2e recipe.

Which denoising steps enter the loss is selected by a **step strategy**
(`--diffusion-step-strategy-path` in `miles/rollout/step_strategy_hub.py`):

| Strategy | Behavior | Typical flags |
|---|---|---|
| `sde_window` | Random contiguous window inside the range (this recipe sets a 10-step window over all 10 steps, so every step is SDE) | `--diffusion-num-sde-steps 10`, `--diffusion-sde-window-range 0,10` |
| `epoch_global_random_choice` | Random subset of candidate steps, shared by each block of `--rollout-batch-size` consecutive prompt groups (one rollout, unless a dynamic-sampling filter drops groups) | `--diffusion-sde-candidate-steps …`, `--diffusion-num-sde-steps` |

Train-side dynamics and more on these strategies: [SDE step backend](/diffusion/advanced/sde-backend).

**DiffusionNFT + PickScore** (3 GPUs, ODE rollouts from the EMA policy) is a separate recipe
covered in the SD3 model guide.

## Inspecting a run

| Question | Where to look |
|---|---|
| Is reward improving? | `rollout/reward/raw_mean` in WandB (`[reward stats] raw mean` in stdout) |
| Train/rollout SDE aligned? | `train/log_prob_mean_abs_diff` (should stay small) |
| Policy loss stable? | `train/loss`, `train/kl_loss` |
| Rollout or train bottleneck? | `perf/rollout_time`, `perf/train_time` |
| Generated images? | WandB `rollout_media/sample_images` |
