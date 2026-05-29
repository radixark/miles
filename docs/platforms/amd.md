---
title: AMD MI300X
description: ROCm 6.3+ with patches for virtual memory management. Same launch scripts.
---

# AMD MI300X

Miles runs on AMD Instinct GPUs (MI300, MI325, MI350, MI355X) with ROCm. The launch
scripts are the same as on NVIDIA — only the container and a few env vars differ.

## Container images

```bash
# MI350 / MI355X
docker pull rlsys/miles:MI350-355-latest

# MI300 / MI325
docker pull rlsys/miles:MI300-latest
```

Or build from the repo:

```bash
cd docker
docker build -f Dockerfile.rocm_MI350-5 -t rlsys/miles:latest .
```

The base ROCm image bundles the patches needed for virtual memory management on MI300X
— thanks to [Yang Wang](https://www.microsoft.com/en-us/research/people/yangwang5/) for
that work.

## Launch the container

```bash
docker run --rm -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  --name miles_dev \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -w $PWD \
  rlsys/miles:latest \
  /bin/bash
```

Inside, install Miles editable:

```bash
git clone https://github.com/radixark/miles.git
cd miles && pip install -e . --no-deps
```

## Download model + data

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17K --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
```

## Convert weights (CPU + Gloo)

We force CPU-only conversion on AMD to bypass some ROCm-specific issues. A GPU-based
ROCm converter is in development.

```bash
cd /root/miles
source scripts/models/qwen3-4B.sh
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-4B \
   --save           /root/Qwen3-4B_torch_dist
```

If you see `miles cannot be found`, re-run `pip install -e . --no-deps` in the repo.

## Launch

The standard `scripts/run-qwen3-4B.sh` works as-is. The image already sets the
ROCm-specific env vars you'd otherwise need:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0   # or 9.4.0 for MI300
NCCL_NET=Socket                   # for non-RDMA setups
NCCL_IB_HCA=...                   # if your fabric supports it
PYTORCH_NO_HIP_MEMORY_CACHING=0
```
