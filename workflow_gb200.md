# Two-Node GB200 Pure-OPD Workflow

This document records the working two-node GB200 workflow for the Qwen3.5-35B-A3B pure-OPD experiment. It describes the exact checkout, mounts, runtime shims, launch sequence, and the reason for each local change.

## 1. Checkout and Branch

Use the dedicated checkout and branch:

```bash
cd /home/scratch.kaixih_ent/repo/miles-opd-gb200-main
git switch opd-gb200-main
```

`opd-gb200-main` contains the four commits from PR #1488 cherry-picked onto the current `origin/main`, followed by the GB200 fixes documented below. Do not confuse it with `/home/scratch.kaixih_ent/repo/miles`, which is a different checkout.

The local `lab/` directory is intentionally not part of the commit. It contains experiment launchers, temporary repros, and credentials used only in the DLCluster workspace.

## 2. Storage and Container Mounts

The same shared storage has three path names:

```text
DLCluster login:  /home/scratch.kaixih_ent
GB compute node:  /mnt/cifs/home/scratch.kaixih_ent
Miles container:  /scratch
```

Launch the ARM Miles image with these bind mounts:

```text
/mnt/cifs/home/scratch.kaixih_ent/repo/miles-opd-gb200-main -> /workspace/miles
/mnt/cifs/home/scratch.kaixih_ent                              -> /scratch
```

The launcher uses:

```bash
MILES_DIR=/workspace/miles
PYTHONPATH=/workspace/miles:/root/Megatron-LM/
```

Therefore Python imports Miles from the mounted checkout, including its committed changes. Megatron, PyTorch, torch-memory-saver, and SGLang still come from `radixark/miles:latest`; their installed source files are not edited.

## 3. Build the Runtime Compatibility Shims

The temporary shim sources are local lab artifacts and are intentionally not committed:

```text
lab/opd_gb200/tms_cumem_trace.c
lab/opd_gb200/torch_shm_unlink_compat.c
```

Compile them on an ARM GB200 compute node, inside the same Miles image. Do not compile them on the x86 login node.

From an active two-node allocation shell:

```bash
cd /home/scratch.kaixih_ent/repo/miles-opd-gb200-main
export ALLOC_JOB_ID="${SLURM_JOB_ID}"
bash lab/opd_gb200/00_build_tms_compat.sh
```

The build script runs the equivalent of:

```bash
mkdir -p /scratch/repro/opd-self-distill/runtime/aarch64

cuda_header=$(find /usr/local/cuda* -path '*/targets/*/include/cuda.h' -print -quit)
cuda_include=$(dirname "${cuda_header}")

cc -shared -fPIC -O2 -I"${cuda_include}" \
  /workspace/miles/lab/opd_gb200/tms_cumem_trace.c \
  -o /scratch/repro/opd-self-distill/runtime/aarch64/libtms_cumem_compat.so \
  -ldl

cc -shared -fPIC -O2 \
  /workspace/miles/lab/opd_gb200/torch_shm_unlink_compat.c \
  -o /scratch/repro/opd-self-distill/runtime/aarch64/libtorch_shm_unlink_compat.so \
  -ldl
```

The persistent host equivalents are:

```text
/home/scratch.kaixih_ent/repro/opd-self-distill/runtime/aarch64/libtms_cumem_compat.so
/home/scratch.kaixih_ent/repro/opd-self-distill/runtime/aarch64/libtorch_shm_unlink_compat.so
```

Verify both outputs before launching:

```bash
file /home/scratch.kaixih_ent/repro/opd-self-distill/runtime/aarch64/*.so
```

They must be AArch64 shared objects. `phase2_gb200.sh` passes their container paths into the Ray runtime environment, and `miles/ray/actor_group.py` prepends them to `LD_PRELOAD` before the original torch-memory-saver library.

## 4. Launch the Two-Node Run

The allocation must contain two `gb200nvl72` nodes with four GPUs per node. The local launcher starts one container per node, creates a Ray head on task 0, joins task 1 as a worker, and submits the Phase-2 payload from `/workspace/miles`.

Run the current accuracy-focused recipe:

```bash
cd /home/scratch.kaixih_ent/repo/miles-opd-gb200-main

RUN_ID="opd-gb200-$(date +%m%d-%H%M)" \
NUM_ROLLOUT=12 \
SKIP_EVAL_BEFORE_TRAIN=1 \
EVAL_INTERVAL=2 \
bash lab/opd_gb200/04_run_long_8k_hybridep.sh
```

The effective settings are:

```text
2 nodes x 4 GB200 GPUs
TP=2, CP=2, EP=8
Flex dispatcher with HybridEP
global batch size=256
max tokens per GPU=8192
CUDA graph capture sizes=1 2 4
checkpoint saving disabled
```

CUDA graph sizes larger than 4 currently use eager decode. This is a conservative mitigation for a separate replay hang after online weight handoff; it is not a final SGLang fix.

## 5. What Changed and Why

### Miles model-input alignment

Files:

```text
miles/backends/megatron_utils/model.py
miles/backends/training_utils/data.py
```

Dynamic THD batches can produce different token-row counts on different EP ranks. HybridEP expects rank-consistent metadata and buffer capacity. A mismatch caused `HYBRID-EP ALLGATHER TIMEOUT` and corrupted updates with extremely large gradient norms and importance-sampling ratios.

For Flex + HybridEP only, Miles now all-reduces the maximum aligned token-row count across the EP group and pads every rank's model input to that common target.

### Miles routing-replay alignment

File:

```text
miles/backends/training_utils/replay_data.py
```

Model inputs and routing replay must use the same EP-wide row target. Local-only replay padding produced errors such as `replay n_tokens ... does not match scores n_tokens ...` under TP2 plus sequence parallelism. Replay rows are now padded to the same EP-wide target before TP/SP slicing; padding remains invalid with value `-1` and stays outside the loss.

### GB200 recipe and runtime environment

File:

```text
examples/on_policy_distillation/qwen3_5_35b_selfdistill/phase2_gb200.sh
```

The script now selects HybridEP explicitly and makes rollout count, batch sizes, response length, token budget, evaluation cadence, checkpoint cadence, and SGLang CUDA graph sizes configurable. It also forwards the two shim paths into Ray actors.

### Ray actor preload composition

File:

```text
miles/ray/actor_group.py
```

The actor environment composes this preload order:

```text
libtorch_shm_unlink_compat.so
libtms_cumem_compat.so
the original torch-memory-saver preload library
```

`libtms_cumem_compat.so` bypasses torch-memory-saver only for small allocations that are not aligned to the GB200 CUDA VMM granularity, routing them to native `cudaMalloc`.

`libtorch_shm_unlink_compat.so` makes cleanup idempotent when PyTorch calls `shm_unlink()` for an already-removed `/torch_*` CUDA IPC refcount file. Only `ENOENT` for `/torch_*` is converted to success.

These two shims are dependency workarounds. The permanent fixes belong in torch-memory-saver/container packaging and PyTorch respectively, not in OPD mathematics.

## 6. Validation Signals

A healthy run must have all of the following:

- no `HYBRID-EP ALLGATHER TIMEOUT`;
- no routing replay row mismatch;
- no fatal CUDA IPC cleanup error;
- `OIS`, `ESS`, and `TIS` remain near 1;
- finite, ordinary-scale gradient norms;
- successful online weight synchronization and the next rollout;
- response length and reverse-KL move toward the teacher behavior.

The validated run changed mean response length from about 18.8k to 6.6k tokens after the first update, reduced truncation from 0.367 to 0.043, and reduced per-token OPD reverse-KL from 0.0448 to 0.0130 while keeping the training diagnostics healthy.

No Megatron, PyTorch, torch-memory-saver, or SGLang installed source file is modified by this workflow. The active code consists of the mounted Miles checkout, the two external runtime shims, and recipe flags.
