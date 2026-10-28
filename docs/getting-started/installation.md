---
title: Installation
description: Install Miles on NVIDIA or AMD GPUs. Docker is the recommended path.
---
There are four ways to install Miles. Docker is recommended because Miles pins patched
versions of SGLang, Megatron-LM, and a few CUDA kernels. pip covers the FSDP and torchtitan
backends, which need none of the patched kernels.

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
  <Tab title="AMD">

    ```bash
    docker pull rocm/sgl-dev:miles-rocm10-mi35x    # or miles-rocm724-mi35x; miles-rocm10-mi30x on MI300X / MI325X

    docker run --rm \
      --device /dev/kfd --device /dev/dri --group-add video --group-add render \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
      --shm-size 128G \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --network=host \
      -it rocm/sgl-dev:miles-rocm10-mi35x /bin/bash
    ```

    See [AMD ROCm](/hardware-platforms/amd-gpus) for more details.

  </Tab>

</Tabs>

The image ships with:

- PyTorch (matching the container's CUDA / ROCm version)
- Megatron-LM, SGLang, FlashAttention-3, DeepGEMM, Apex
- Ray, uv, and Miles installed editable at `/root/miles`

See [Hardware requirements](#hardware-requirements) for per-GPU status.

## Method 2: pip (FSDP and torchtitan)

Every tagged release from v0.1.4 on publishes a pure-Python `miles` wheel, and every commit of the
`sglang-miles` branch publishes an `sglang` wheel, to a pip index at
`https://radixark.github.io/miles-wheels/simple/`. Linux x86_64 or aarch64, Python 3.11 or 3.12.

PyPI has an unrelated project named `miles`, and pip gives every index equal weight, so install
with uv, which takes each package from the first index that has it:

```bash
uv pip install --index https://radixark.github.io/miles-wheels/simple/ \
  "miles[torchtitan]==<version>" "sglang==<sglang version>"
```

With plain pip, install the wheel by URL instead, with the index only for `sglang`:

```bash
pip install "miles[torchtitan] @ https://github.com/radixark/miles/releases/download/v<version>/miles-<version>-py3-none-any.whl" \
  "sglang==<sglang version>" --extra-index-url https://radixark.github.io/miles-wheels/simple/
```

- The release's notes name the `sglang` build it was tested with (`<tag>.post<n>+miles.g<commit>`,
  the `sglang-miles` commit the wheel came from). Pin that one.
- `[torchtitan]` adds torchtitan and its torch floor; leave it off for FSDP alone. torch itself
  follows SGLang's pin.
- Megatron-LM is not installable this way: `--train-backend megatron` needs the Docker image.
  True-on-policy sampling-support replay also needs the image for now (it relies on
  FlashAttention-3 and the Miles router build).

Training still launches from a checkout's `train.py` ([Quick Start](/getting-started/quick-start));
the wheel supplies the `miles` package it imports.

## Method 3: From source

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

## Method 4: Update an existing container

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

If either command fails, see [Debugging](/developer/debug).

## Hardware requirements

| Hardware | Status |
|---|---|
| NVIDIA GB300 / GB200 / B300 / B200 | Production |
| NVIDIA H200 / H100 | Production (CI guarded) |
| NVIDIA A100 | Supported — FP8 features disabled |
| AMD MI300X, MI325, MI350X, MI355X | Supported via ROCm |

For multi-node training you also need a high-bandwidth interconnect — InfiniBand, RoCEv2,
or Slingshot — and 200+ GB/s per node. Single-node jobs run fine over NVLink only.

## Next steps

- [Quick Start](/getting-started/quick-start) — run your first training job.
- [Core concepts](/user-guide/concepts) — the mental model behind Miles.
- [Training backends](/user-guide/training-backend) — Megatron vs FSDP.
