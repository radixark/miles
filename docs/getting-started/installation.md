---
title: Installation
description: Install Miles on NVIDIA or AMD GPUs. Docker is the recommended path.
---

# Installation

There are three ways to install Miles. Docker is recommended because Miles pins patched
versions of SGLang, Megatron-LM, and a few CUDA kernels.

## Method 1: Docker (recommended)

<Tabs>

  <Tab title="NVIDIA">

    ```bash
    docker pull radixark/miles:latest

    docker run --rm \
      --gpus all --ipc=host --shm-size=32g \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --network=host \
      -it radixark/miles:latest /bin/bash
    ```

  </Tab>
  <Tab title="AMD MI300X / MI350X">

    ```bash
    docker pull rlsys/miles:MI350-355-latest    # or MI300-latest

    docker run --rm \
      --device /dev/dri --device /dev/kfd \
      --group-add video --ipc=host --shm-size=32g \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
      --privileged \
      -it rlsys/miles:MI350-355-latest /bin/bash
    ```

  </Tab>

</Tabs>

The image ships with:

- PyTorch (matching the container's CUDA / ROCm version)
- Megatron-LM, SGLang, FlashAttention-3, DeepGEMM, Apex
- Ray, uv, and Miles installed editable at `/root/miles`

See [Platforms](../platforms/index.md) for platform-specific notes.

## Method 2: From source

Clone and install in an existing environment.

```bash
git clone https://github.com/radixark/miles.git
cd miles
pip install -r requirements.txt
pip install -e . --no-deps
```

<Warning>

**Patched dependencies.** Miles pins patched versions of SGLang and Megatron-LM. Installing them yourself at
the wrong commit is the most common source of bug reports — use Docker if you can.

</Warning>

## Method 3: Update an existing container

If you already run a Miles image and want the latest code:

```bash
cd /root/miles
git pull --rebase
pip install -e . --no-deps
ray stop && ray start --head --port=6379
```

## Verify

Confirm Miles imports and the GPUs are visible:

```bash
python -c "import miles; print('Miles import OK')"
nvidia-smi
```

If either command fails, see [Debugging](../developer/debug.md) or the [FAQ](../faq.md).

## Hardware requirements

| Hardware | Status |
|---|---|
| NVIDIA H100 / H200 | Production (CI guarded) |
| NVIDIA B100 / B200 | Production |
| NVIDIA A100 | Supported — FP8 features disabled |
| AMD MI300X, MI325, MI350X, MI355X | Supported via ROCm |

For multi-node training you also need a high-bandwidth interconnect — InfiniBand, RoCEv2,
or Slingshot — and 200+ GB/s per node. Single-node jobs run fine over NVLink only.

## Next steps

- [Quick Start](quick-start.md) — run your first training job.
- [Core concepts](../user-guide/concepts.md) — the mental model behind Miles.
- [Training backends](../user-guide/usage.md) — Megatron vs FSDP.
