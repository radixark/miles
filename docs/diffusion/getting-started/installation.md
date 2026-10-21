---
title: Installation
description: Get a working miles-diffusion environment — Docker (recommended) or from source.
---
miles-diffusion trains a diffusion DiT with FSDP2 while
[sglang-diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen) serves the rollout.
The two halves must agree numerically, so the pinned versions matter more than usual: the rollout engine tracks
**sglang main**, not a release tag, and the training side pins **torch 2.11.0** (an FSDP monkey patch is version-gated
on it). A future update will upgrade the training stack to **torch 2.13**.

Use Docker unless you have a reason not to.

## Method 1: Docker (recommended)

```bash
docker pull radixark/miles_diffusion:latest
```

`latest` tracks sglang main and is rebuilt every few days; dated tags
(`dev-cu129-sglang-main-YYYYMMDD`) pin a specific build. CUDA 12.9 is the only supported
build — the Dockerfile carries a CUDA 13 recipe in a comment, but its `sglang-kernel` pin is
hardcoded to cu129 and no CI covers it.

### Build it yourself

```bash
git clone https://github.com/radixark/miles_diffusion.git
cd miles_diffusion
docker build -f docker/Dockerfile -t miles-diffusion:$(cat docker/version.txt) .
```

Useful build args:

| Arg | Default | What it does |
|---|---|---|
| `SGLANG_IMAGE_TAG` | `v0.5.12-cu129` | Base `lmsysorg/sglang` image. |
| `SGLANG_DIFFUSION_BRANCH` | `sglang-miles-h3` | sglang branch the rollout engine is built from; carries upstream `main` plus the diffusion rollouts this repo needs ahead of it. |
| `SGLANG_DIFFUSION_COMMIT` | `none` | Pin a sglang sha; `none` follows the branch tip. |
| `FA3_WHEELS_TAG` | `cu129-x86_64` | Which prebuilt FlashAttention-3 wheel to pull. |
| `MILES_DIFFUSION_COMMIT` | `main` | Ref of miles_diffusion baked into the image. |

### Run

```bash
docker run --rm \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v /your/datasets:/root/datasets \
  -e HF_TOKEN=$HF_TOKEN \
  -it radixark/miles_diffusion:latest /bin/bash
```

The image ships with:

- PyTorch 2.11.0 and the sglang base image's CUDA stack
- sglang built from main (`/sgl-workspace/sglang`, editable) with `sglang.multimodal_gen`
- FlashAttention-3 (`flash_attn_interface`), `sglang-kernel==0.4.5`, `torch_memory_saver==0.0.9`
- `diffusers`, `peft`, `transformers`, `ray`, `wandb`, and `ltx-core` from `requirements.txt`
- miles_diffusion installed editable at `/root/miles_diffusion`
- PaddleOCR's English det/rec/cls weights pre-downloaded (the OCR reward would otherwise
  race-download them at runtime)
- `nccl-tests` binaries on `PATH` for link diagnostics

To run your own working tree instead of the baked copy, bind-mount it and reinstall:

```bash
docker run ... -v $PWD:/root/miles_diffusion -it radixark/miles_diffusion:latest /bin/bash
cd /root/miles_diffusion && pip install -e . --no-deps
```

## Method 2: Update an existing container

If you already run the image and want the latest code:

```bash
cd /root/miles_diffusion
git pull --rebase
pip install -e . --no-deps
```

No Ray restart is needed — the launch scripts stop and restart the cluster themselves.

## Method 3: Bare metal, from source

<Warning>

**Strongly discouraged unless the image genuinely cannot run on your machine.** Unlike
Docker, this mutates the host: it installs apt packages and replays the image's pinned
package set into the **system** Python with `--no-deps`, overwriting whatever versions are
already there. Do not run it on a machine you use for anything else.

</Warning>

```bash
apt-get update && apt-get install -y --no-install-recommends git ca-certificates
git clone https://github.com/radixark/miles_diffusion.git
cd miles_diffusion
bash .claude/skills/install-miles-diffusion/install.sh
```

Written against a CUDA 12.9 / Ubuntu 24.04 host, to match the image. Nothing enforces that,
but the apt versions are unpinned, so another release hands out different versions of the
apt-sourced dists and the verify step reports drift.

The installer has six idempotent steps, and `--from STEP` resumes a failed run. The last step diffs every installed
package against the image's snapshot — a clean run ends with:

```
matched 369   mismatched 0   missing 0   expected-absent 4   extra 0
environment matches the official image
```

To check an existing machine against the image without installing anything:
`python3 .claude/skills/install-miles-diffusion/verify_env.py`.

Plain `pip install -r requirements.txt && pip install -e . --no-deps` installs, but leaves a
**trainer-only** environment: CPU tests and `--train-only` SFT run, RL does not —
`sglang.multimodal_gen` and FlashAttention-3 are not on PyPI. Resolving the full set through
pip is not possible (it is only consistent under `--no-deps` replay), which is what the
installer handles.

## Verify

```bash
# package imports
python -c "import miles; print('miles-diffusion import OK')"

# rollout engine is present
python -c "from sglang.multimodal_gen.runtime.server_args import ServerArgs; print('sglang-d OK')"

# GPUs visible
nvidia-smi

# arguments parse
python3 train_diffusion.py --help | head
```

Then run the CPU test suite — seconds, and it catches most environment breakage:

```bash
pytest tests/fast -x -q
```

On a machine with no sglang GPU kernels installed, install the CPU stubs first, the way CI does:
`uv pip install tests/ci/cpu_stubs`.

## Hardware

| | |
|---|---|
| Validated GPUs | H200 — what the CI runners are, and what every shipped recipe was tuned on. |
| Minimum for a real run | 2 GPUs — SD3.5-medium LoRA GRPO colocated (`scripts/run_diffusion_grpo_sd3_ocr_sglang.py`) |
| Typical | 4 train GPUs + 1 dedicated reward GPU (the Qwen-Image / Wan2.2 / LTX recipes) |

Under `--colocate` the training actor and the rollout engines time-share the same GPUs,
so the floor is set by whichever of the two needs more memory, not by their sum.

## Environment variables

| Variable | When you need it |
|---|---|
| `HF_TOKEN` | Gated checkpoints. SD3.5 needs it **even when the weights are cached** — sglang still fetches `model_index.json` from the hub at startup. |
| `WANDB_API_KEY` | Without it the launch scripts silently drop all `--use-wandb` flags. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Set by every launch script; reduces fragmentation OOMs. |
| `MILES_DIFFUSION_MODEL_FAMILY` | Escape hatch when `--diffusion-model-family` is unset. It overrides checkpoint-name auto-detection, but an explicit CLI family takes precedence. |

## Next

- [Launch Scripts](/diffusion/user-guide/launch-script) — what a launch script does and how to override a
  recipe.
- [CLI Reference](/diffusion/user-guide/cli-reference) — every flag.
