---
title: GLM4.5
description: Launch recipe for GLM4.5 106B-A12B and 355B-A32B — multi-node MoE with R3 and unified FP8.
---

# GLM4.5

GLM4.5 is Zhipu's MoE line: 106 B-A12B for two-node experimentation and 355 B-A32B for eight-node frontier runs. Miles's canonical recipe enables R3 (Rollout Routing Replay) and unified FP8 by default, and the 355 B can also run under INT4 QAT to fit on a single H200 node.

## Variants

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| GLM4.5-106B-A12B | 12 B / 106 B | `zai-org/GLM4.5-106B-A12B` | `scripts/models/glm4.5-106B-A12B.sh` |
| GLM4.5-355B-A32B | 32 B / 355 B | `zai-org/GLM-4.5` | `scripts/models/glm4.5-355B-A32B.sh` |

## Quick start (GLM4.5-355B-A32B, 8 nodes × 8 H100)

```bash
cd /root/miles

# 1. Env (BASE_DIR must be on a shared FS reachable from every node)
export BASE_DIR=/shared/glm4-5
export MASTER_ADDR=<head node IP>

# 2. Download weights once on the shared FS
hf download zai-org/GLM-4.5 --local-dir $BASE_DIR/GLM-4.5-355B-A32B

# 3. Convert HF → Megatron dist (run on each of the 2 nodes with NODE_RANK=0..1)
source scripts/models/glm4.5-355B-A32B.sh
PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=2 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/GLM-4.5-355B-A32B \
   --save          $BASE_DIR/GLM-4.5-355B-A32B_torch_dist

# 4. Launch — node 0 runs the script; other nodes attach to the Ray cluster
bash scripts/run-glm4.5-355B-A32B.sh
```

On every non-head node, attach to the Ray cluster:

```bash
ray start --address=${MASTER_ADDR}:6379 \
          --num-gpus 8 \
          --node-ip-address ${WORKER_IP} \
          --disable-usage-stats
```

## Expected signal

Standard `loss=… reward=…` trainer stdout. MoE-specific signals:

- `router/replay_hit_rate` climbs to ≈ 1.0 quickly once R3 is engaged.
- FP8 runs keep trainer-step time within ~15% of BF16 at 355 B scale; larger gaps usually indicate `--fp8-amax-history-len` is too short or `--sglang-fp8-quantization-method` is misconfigured.

---

## Deep dive

### Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-glm4.5-355B-A32B.sh` | 64× H100 | 8 nodes × 8 GPUs |
| `scripts/run_glm45_355b_a32b.py` | — | Programmatic entrypoint |

??? tip "Fan-out from node 0"
    Append this loop at the end of the head launch script and the head will SSH out to workers:

    ```bash
    for WORKER_IP in $(awk '{print $1}' $BASE_DIR/mpi_hostfile); do
      [[ "$WORKER_IP" == "$MASTER_ADDR" ]] && continue
      ssh root@"${WORKER_IP}" \
        "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
         ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 \
                   --node-ip-address ${WORKER_IP} --disable-usage-stats" &
    done
    wait
    ```

### Parallelism

| Model | TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 106B-A12B | 4 | 8 | 2 | 2 | 16 384 | 16 |
| 355B-A32B | 8 | 16 | 4 | 2 | 16 384 | 64 |

### Unified FP8 + R3 (recommended for production)

```bash
GRPO_ARGS+=(
   --use-miles-router
   --use-rollout-routing-replay
   --use-tis
)
SGLANG_ARGS+=(
   --sglang-use-miles-router
   --sglang-fp8-quantization-method per-tensor
)
PERF_ARGS+=(
   --fp8-format hybrid
   --fp8-amax-history-len 1024
)
```

See [FP8 & Low Precision](../../advanced/fp8-low-precision.md).

### INT4 QAT alternative

To fit the 355 B variant on a single H200 node, see [INT4 QAT](../../advanced/int4-qat.md).

### Tuning knobs

| Symptom | Fix |
|---|---|
| NCCL hang during weight sync | Set `NCCL_TIMEOUT=900`, enable `NCCL_DEBUG=INFO` |
| Expert imbalance | Confirm R3 is on |
| OOM mid-train | Drop `max-tokens-per-gpu` to 12 288 |
| Checkpoint write saturating GPFS | Raise `--save-interval`, switch FS |
