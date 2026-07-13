# Two-Node GB200 Pure-OPD Workflow

Validated on 2026-07-13 with two independent two-node runs. Each run completed
two real rollouts and two online training updates with overlap scheduling,
CUDA graph padding, and power-of-two capture enabled.

## Required Stack

Use these pieces together:

1. [Miles PR #1634](https://github.com/radixark/miles/pull/1634) from
   branch `agent/gb200-hybridep-fixes`.
2. The narrow PyTorch `shm_unlink(ENOENT)` compatibility shim. This is the
   only remaining local workaround without an upstream PR.
3. A real installation of
   [torch-memory-saver PR #82](https://github.com/fzyzcjy/torch_memory_saver/pull/82)
   at commit
   `c96bf60e093b4bec2b045cb5b8a08601d0ae8a79`.
4. [SGLang PR #27140](https://github.com/sgl-project/sglang/pull/27140):
   invalidate stale CUDA graphs and recapture them after torch-memory-saver
   resumes graph-owned memory.
5. [SGLang PR #31073](https://github.com/sgl-project/sglang/pull/31073):
   synchronize asynchronous device work before torch-memory-saver unmaps its
   backing memory.
6. [SGLang PR #31072](https://github.com/sgl-project/sglang/pull/31072): for
   hybrid linear-attention models, publish the overlap-scheduler read-done
   event after CUDA graph replay, not before it.
7. Capture exactly `1 2 4 8 16 32 64 128 256 512`, with graph padding and
   overlap scheduling enabled.

SGLang PRs #30895 and #30974 are not required for this hang. The second
validation completed with both disabled. They may remain useful as padded-row
hygiene, but they are not part of the minimum validated stack. The other
padding, replay-debug, and FlashInfer patches under `lab/opd_gb200/patches/`
are disabled experiments and are not part of this workflow.

Do not use the historical `libtms_cumem_compat.so` shim. Keep
`TMS_CUMEM_TRACE_LIB` empty. The local `lab/` directory contains launchers and
temporary patches and is intentionally not committed.

## 1. Get Two Nodes

Run this from the visible tmux shell on `dl3`:

```bash
salloc_node gb200nvl72 2
```

The launcher uses one Slurm task and four GPUs per node.

## 2. Check the Checkouts and Mounts

```text
Miles PR #1634:  /home/scratch.kaixih_ent/repo/miles-hybridep-pr
Launcher/lab:    /home/scratch.kaixih_ent/repo/miles-opd-gb200-main
TMS PR #82:      /home/scratch.kaixih_ent/repo/torch_memory_saver
```

The compute-node mounts are:

```text
/mnt/cifs/home/scratch.kaixih_ent/repo/miles-hybridep-pr -> /workspace/miles
/mnt/cifs/home/scratch.kaixih_ent/repo/miles-opd-gb200-main/lab -> /workspace/lab
/mnt/cifs/home/scratch.kaixih_ent/repo/torch_memory_saver -> /workspace/torch_memory_saver
/mnt/cifs/home/scratch.kaixih_ent -> /scratch
```

## 3. Build the Shim and Install TMS

Build the AArch64 `shm_unlink` shim inside the Miles container on a GB200 node:

```bash
cd /home/scratch.kaixih_ent/repo/miles-opd-gb200-main
export ALLOC_JOB_ID="${SLURM_JOB_ID}"
bash lab/opd_gb200/00_build_tms_compat.sh
```

Only this runtime shim is loaded:

```text
/scratch/repro/opd-self-distill/runtime/aarch64/libtorch_shm_unlink_compat.so
```

`node_entrypoint.sh` copies the mounted TMS source to node-local storage, runs
`make reinstall`, and verifies the installed PR #82 library. Both nodes must
print:

```text
TMS_PR82_VERIFIED=1
TMS_SOURCE_COMMIT=c96bf60e093b4bec2b045cb5b8a08601d0ae8a79
```

## 4. Launch a Two-Update Validation

Until the SGLang PRs are present in the container image, the launcher applies
these exact backports:

| Upstream PR | Launcher patch | Switch |
| --- | --- | --- |
| [#27140](https://github.com/sgl-project/sglang/pull/27140) | `sglang_pr27140_forward_port.patch` and `sglang_pr27140_graph_reset.patch` | `SGLANG_RECAPTURE_AFTER_WEIGHT_UPDATE=pr27140` |
| [#31073](https://github.com/sgl-project/sglang/pull/31073) | `sglang_sync_before_tms_pause.patch` | `SGLANG_SYNC_BEFORE_MEMORY_RELEASE=1` |
| [#31072](https://github.com/sgl-project/sglang/pull/31072) | `sglang_hybrid_post_replay_war.patch` | `SGLANG_HYBRID_POST_REPLAY_WAR=1` |

Do not add an unnamed source patch to this workflow. Create or reference its
upstream PR first.

```bash
cd /home/scratch.kaixih_ent/repo/miles-opd-gb200-main

RUN_ID="opd-p2-hybridwar-$(date +%m%d-%H%M)" \
HOST_MILES=/home/scratch.kaixih_ent/repo/miles-hybridep-pr \
NUM_ROLLOUT=2 \
SKIP_EVAL_BEFORE_TRAIN=1 \
EVAL_INTERVAL=5 \
SAVE_INTERVAL=none \
ROLLOUT_BATCH_SIZE=32 \
N_SAMPLES_PER_PROMPT=8 \
OVER_SAMPLING_BATCH_SIZE=32 \
GLOBAL_BATCH_SIZE=256 \
MAX_TOKENS_PER_GPU=8192 \
MOE_FLEX_DISPATCHER_BACKEND=hybridep \
SGLANG_CUDA_GRAPH_BS="1 2 4 8 16 32 64 128 256 512" \
SGLANG_RECAPTURE_AFTER_WEIGHT_UPDATE=pr27140 \
SGLANG_SYNC_BEFORE_MEMORY_RELEASE=1 \
SGLANG_HYBRID_POST_REPLAY_WAR=1 \
SGLANG_PR30895=0 \
SGLANG_PR30974=0 \
SGLANG_DISABLE_CUDA_GRAPH_PADDING=0 \
SGLANG_NEUTRALIZE_MOE_CUDA_GRAPH_PADDING=0 \
FLASHINFER_DISTRIBUTED_AUTOTUNE_SYNC=0 \
TMS_CUMEM_TRACE_LIB="" \
TORCH_SHM_UNLINK_COMPAT_LIB=/scratch/repro/opd-self-distill/runtime/aarch64/libtorch_shm_unlink_compat.so \
bash lab/opd_gb200/01_launch_phase2_pure.sh
```

For a B200-comparable curve, set `NUM_ROLLOUT=12` and
`SKIP_EVAL_BEFORE_TRAIN=0`; keep the runtime stack and capture list unchanged.

## 5. Validate the Result

A successful run must show:

- `TMS_PR82_VERIFIED=1` on both nodes;
- the #27140 recapture and #31073 pre-release synchronization backports
  verified;
- `SGLANG_HYBRID_POST_REPLAY_WAR_VERIFIED=1` on both nodes;
- `disable_overlap_schedule=False` and `disable_cuda_graph_padding=False`;
- all ten power-of-two graph buckets recaptured after weight updates;
- second-rollout padded batches reporting `cuda graph: True`;
- no HybridEP timeout, CUDA IPC error, replay error, or non-finite metric;
- OIS, ESS, and TIS near 1 and a finite ordinary-scale gradient norm;
- Ray reporting `Job 'qwen3.5-opd-pure' succeeded`.

## Validation Evidence

Both runs below used the new overlap-event ordering and completed two real
rollout/update cycles:

| Run | Old padding PRs | Response length | Reverse-KL | Gradient norm |
| --- | --- | --- | --- | --- |
| `opd-p2-hybridwar-r1c-0713-1937` | #30895/#30974 on | 18881.8 -> 6469.1 | 0.04488 -> 0.01315 | 0.3104 -> 0.1387 |
| `opd-p2-hybridwar-no56-r1c-0713-2041` | #30895/#30974 off | 18931.8 -> 6551.2 | 0.04466 -> 0.01359 | 0.3112 -> 0.1376 |

For the A/B run with #30895/#30974 disabled, update 1 finished with
`OIS=1.0000001`, `ESS=0.9999997`, and `TIS=1.0000067`. Its second rollout
crossed padded request counts including 78, 75, 73, 70, and 69 while remaining
on CUDA graph replay.

The event-order unit test also passes in the Miles container:

```text
plain attention: load -> read-done event -> replay
hybrid attention: load -> replay -> read-done event
2 passed
```

## Why Each Change Exists

- **Miles #1634:** keeps model-input and routing-replay token rows aligned
  across HybridEP ranks.
- **PyTorch unlink shim:** makes cleanup idempotent only for an already-removed
  `/torch_*` CUDA-IPC refcount file.
- **TMS #82:** fixes the GB200 CUDA VMM allocation-size failure.
- **SGLang #27140:** prevents replaying graph objects whose memory was released
  and then restored by TMS.
- **SGLang #31073:** finishes asynchronous device work before TMS unmaps its
  backing memory.
- **SGLang #31072:** the existing WAR fast path publishes read-done
  after `load_batch`, which is safe for plain attention. Hybrid/Mamba decode
  keeps reading shared request/state buffers during graph replay, so the
  scheduler could mutate the next iteration's buffers too early. Publishing
  the event after replay closes that race while preserving the fast path for
  plain attention.

At teardown, W&B may emit an `atexit` `BrokenPipeError`. Treat it as teardown
noise only when both updates completed and Ray explicitly reported success.
