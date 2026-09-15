---
title: AMD ROCm
description: Run Miles on AMD MI350X / MI355X and MI300X / MI325X with the ROCm images. Docker is the recommended path.
---
Miles runs on AMD GPUs through ROCm. The ROCm images ship SGLang, Megatron-LM, and Miles
preinstalled, with `MILES_HARDWARE_PLATFORM=rocm` already set. The recipes and `train.py`
flags are the same as on NVIDIA; what changes is the image, the `docker run` flags, and the
launcher path.

## Images

The two `mi35x` images are built daily from `main` by the sgl-project/sglang nightly
workflows and published to Docker Hub under
[`rocm/sgl-dev`](https://hub.docker.com/r/rocm/sgl-dev/tags?name=miles):

| Image | ROCm | GPUs | Notes |
|---|---|---|---|
| `rocm/sgl-dev:miles-rocm720-mi35x` | 7.2 | MI350X / MI355X | Python 3.10 — the image the ROCm CI runs |
| `rocm/sgl-dev:miles-rocm10-mi35x` | 10 | MI350X / MI355X | Python 3.12 |
| `rocm/sgl-dev:miles-rocm700-mi30x` | 7.0 | MI300X / MI325X | Not rebuilt daily — last built 2026-09-08 |

Each undated tag moves with every build; append `-YYYYMMDD` (e.g.
`miles-rocm720-mi35x-20260910`) to pin one. The recipes below are validated on
MI350X / MI355X (`gfx950`); the Qwen3 launchers accept only `MI350X` / `MI355X` in
`--hardware` and do not run on MI300X / MI325X as shipped.

To build an image yourself, `docker/Dockerfile.rocm` holds the recipe:

```bash
python docker/build.py --variant rocm720-mi35x --image-tag dev    # or rocm10-mi35x
```

`dev` writes the undated tag plus a `-YYYYMMDDHHMM` sibling.

## Start the container

On the **host**:

```bash
docker pull rocm/sgl-dev:miles-rocm720-mi35x

docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  --shm-size 128G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -it rocm/sgl-dev:miles-rocm720-mi35x /bin/bash
```

That drops you into a shell inside the container, with Miles at `/root/miles`, Megatron-LM
at `/root/Megatron-LM`, and SGLang at `/sgl-workspace/sglang`.

**Everything from here on runs inside the container.**

## Verify

Confirm Miles imports and the GPUs are visible:

```bash
python -c "import miles; print('Miles import OK')"
rocm-smi --showproductname
```

If either command fails, see [Debugging](/developer/debug).

## Launch training

The AMD launchers live under `scripts/amd/` and mirror the CUDA recipes. Download the model
and data and convert the checkpoint as in the [Quick Start](/getting-started/quick-start)
(Steps 2 and 3), then launch:

```bash
cd /root/miles
python scripts/amd/run_qwen3_4b.py --hardware MI355X    # or MI350X
```

On the two Qwen3 launchers, `--hardware` sets the GPU count per node and defaults to
auto-detection, and the launcher exports the Ray HIP visibility variables so Ray and PyTorch
agree on the device list; `run_qwen3_30b_a3b.py` also takes `--train-fp8` and `--rollout-fp8`.
The DeepSeek-V4, GLM-5.2, and Inkling launchers take `--num-nodes` and `--num-gpus-per-node`
and run as subcommands:

```bash
python scripts/amd/run_deepseek_v4.py train --model-name DeepSeek-V4-Flash-FP8 \
  --num-nodes 4 --num-gpus-per-node 8
```

The CUDA launchers under `scripts/` do not accept `MI3xx` in `--hardware`; use the
`scripts/amd/` counterpart.

| Model | Launcher | Verified on |
|---|---|---|
| Qwen3-4B | `scripts/amd/run_qwen3_4b.py` | 1 node × 8 MI350X / MI355X |
| Qwen3-30B-A3B | `scripts/amd/run_qwen3_30b_a3b.py` | 1 or 2 nodes × 8 MI350X / MI355X |
| DeepSeek-V4-Flash-FP8 | `scripts/amd/run_deepseek_v4.py` | 4 nodes × 8 MI355X, FP8 block-wise training |
| Inkling-Small (4-layer slice) | `scripts/amd/run_inkling.py` | 4 GPUs, CI smoke test |
| GLM-5.2 (5-layer slice) | `scripts/amd/run_glm5_2_744b_a40b.py` | 4 GPUs — ROCm CI test registered but disabled |

Training a different model? See [Models](/models/index) for the per-model recipes; the
launcher flags carry over, the `--hardware` presets do not.

## CI

Adding the `run-ci-amd` label to a pull request runs the ROCm tests registered with the
`amd` label in `stage-c-4-gpu-mi350` on the 4-GPU MI350 runners, against
`rocm/sgl-dev:miles-rocm720-mi35x`; any other `run-ci-<label>` selects the ROCm tests
carrying that label. The `nightly-stage-c-*-mi350` suites run only in the sgl-project/sglang
nightly. The daily image builds and MI355X test runs are tracked in
[`radixark/miles#2256`](https://github.com/radixark/miles/issues/2256); the AMD roadmap is
[`radixark/miles#2025`](https://github.com/radixark/miles/issues/2025).

## Next steps

- [Quick Start](/getting-started/quick-start) — the same Qwen3-4B run, step by step.
- [Hardware requirements](/getting-started/installation#hardware-requirements) — per-GPU status.
- [Low Precision RL](/advanced/low-precision) — FP8 block-wise on MI350X / MI355X.
- [Docker build](/developer/ci/02-docker-build) — the ROCm Dockerfile, variants, and tags.
