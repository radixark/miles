# DeepEP FP8 (MoE / mori) setup and test

This document describes how to launch the ROCm **miles** image, apply **aiter** / **sglang** patches shipped in this repo (optional but recommended for mori EP FP8), install **mori** and **uccl** with the **deep_ep** interface from source (ROCm / MI355X-oriented paths), then run the DeepEP FP8 e2e test in this repo.

Paths like `/sgl-workspace/mori` and `/workspace` match a typical SGLang-style container layout; adjust bind mounts and host paths to your machine.

---

## 1) Launch container

On the host, bind your local **`miles`** checkout into the container (adjust the host path in `-v` as needed):

```bash
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add 44 \
  --group-add 109 \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --memory=0 \
  --memory-swap=0 \
  --privileged \
  --ulimit nofile=65535:65535 \
  -v /tmp:/tmp \
  -v /your_miles_path/miles:/workspace/miles \
  rlsys/miles:MI350-355-latest
```

Inside the container, the repo is typically at **`/workspace/miles`**. Subsequent steps assume you are in that environment (or equivalent paths).

---

## 2) Apply patches (dry-run first)

Do this **inside the container**, after launching it (step 1). Patch files live under **`patches/`** in the **miles** checkout; use a shell whose working directory is the miles repo root so **`$PWD/patches/...`** resolves (e.g. `cd /workspace/miles`).

### 2.1 Dry-run (recommended)

```bash
git -C /sgl-workspace/aiter  apply --check "$PWD/patches/aiter.patch"
git -C /sgl-workspace/sglang apply --check "$PWD/patches/sglang.patch"
```

If **both** commands succeed with no output, the patches match your tree.

### 2.2 Apply

```bash
git -C /sgl-workspace/aiter  apply "$PWD/patches/aiter.patch"
git -C /sgl-workspace/sglang apply "$PWD/patches/sglang.patch"
```

---

## 3) Install mori from source (editable)

```bash
# /sgl-workspace/mori is the editable install (mori.egg-link → python/)
cd /sgl-workspace/mori
git fetch origin
git checkout origin/main             # e94694c7, 2026-06-04 — fix(io): bind worker threads within allowed cpuset
rm -rf build/
MORI_GPU_ARCHS=gfx950 pip install --no-build-isolation -v -e .
```

---

## 4) Install uccl with deep interface (deep_ep)

**Important:** `uccl/ep/install_deps.sh` was **not** run. On ROCm it reinstalls PyTorch nightly (`--index-url .../nightly/rocm7.0`), which would overwrite the image’s customized ROCm torch. Only the minimal build dependencies below were added.

```bash
# 0. Clone uccl
cd /workspace
git clone https://github.com/uccl-project/uccl.git

# 1. Minimal build deps (does not touch torch)
pip install nanobind                       # was missing; setup.py hard-requires it
# libibverbs-dev / libnl-3-dev / libnl-route-3-dev already present in image

# 2. Build & install uccl.ep for MI355X (gfx950) from source against the image's torch
cd /workspace/uccl/ep
TORCH_CUDA_ARCH_LIST=gfx950 PYTORCH_ROCM_ARCH=gfx950 python setup.py install

# 3. Install the drop-in deep_ep package
# (skip dep resolution so PyPI 'uccl' can't override the local build)
cd deep_ep_wrapper
pip install --no-deps --no-build-isolation -e .

# Verify
python -c "import uccl.ep; from deep_ep import Buffer; print('OK')"
```

---

## 5) Run the test

From the **`miles` repository root** in the container (e.g. `/workspace/miles`, the directory that contains `tests/`):

```bash
bash tests/e2e/megatron/test_qwen3_30B_A3B/run_test_deepep_fp8.sh
```

The script sets `PYTHONPATH`, mori / AITER-related env vars, stops any stale Ray cluster, and runs `tests/e2e/megatron/test_qwen3_30B_A3B/test_deepep_fp8.py`.
