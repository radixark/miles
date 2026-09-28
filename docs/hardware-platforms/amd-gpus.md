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
| `rocm/sgl-dev:miles-rocm10-mi35x` | 10 | MI350X / MI355X | Python 3.12 — the image the nightly tests run on |
| `rocm/sgl-dev:miles-rocm724-mi35x` | 7.2.4 | MI350X / MI355X | Python 3.12 |
| `rocm/sgl-dev:miles-rocm700-mi30x` | 7.0 | MI300X / MI325X | Not rebuilt daily — last built 2026-09-08 |

Each undated tag moves with every build; append `-YYYYMMDD` (e.g.
`miles-rocm10-mi35x-20260916`) to pin one.

To build an image yourself, `docker/Dockerfile.rocm` holds the recipe:

```bash
python docker/build.py --variant rocm10-mi35x --image-tag dev    # or rocm724-mi35x
```

## Start the container

On the **host**:

```bash
docker pull rocm/sgl-dev:miles-rocm10-mi35x

docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  --shm-size 128G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -it rocm/sgl-dev:miles-rocm10-mi35x /bin/bash
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

The other recipes under `scripts/amd/` launch the same way.

## Next steps

- [Quick Start](/getting-started/quick-start) — the same Qwen3-4B run, step by step.
- [Hardware requirements](/getting-started/installation#hardware-requirements) — per-GPU status.
- [Low Precision RL](/advanced/low-precision) — FP8 block-wise on MI350X / MI355X.
- [Docker build](/developer/ci/02-docker-build) — the ROCm Dockerfile, variants, and tags.
