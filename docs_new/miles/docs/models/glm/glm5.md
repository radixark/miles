---
title: GLM5
description: Launch recipe for GLM5 744B-A40B — Zhipu's frontier-scale MoE (16 nodes, 40 B active).
---

# GLM5 744B-A40B

GLM5 is Zhipu's frontier-scale MoE: 744 B total parameters, 40 B active per token, 256 routed experts. Miles ships a 16-node launcher plus 4-layer / 20-layer smoke-test configs for iterating on code changes before paying the 16-node bill.

## Architecture

| Property | Value |
|---|---|
| Total params | 744 B |
| Active per token | 40 B |
| Routed experts | 256 |
| Active experts | 8 |
| Shared experts | 1 |
| MoE layers | 75 (3 dense heads) |
| Hidden size | 6144 |
| Heads | 64 |

Full config in `scripts/models/glm5-744B-A40B.sh`. Short test variants (`glm5-744B-A40B_4layer.sh`, `glm5-744B-A40B_20layer.sh`) are available for quick smoke tests.

## Quick start (16 nodes × 8 H100)

```bash
cd /root/miles

# 1. Env (shared FS required)
export BASE_DIR=/shared/glm5
export MASTER_ADDR=<head node IP>

# 2. Download weights once on shared FS
hf download zai-org/GLM5-744B-A40B --local-dir $BASE_DIR/GLM5-744B-A40B

# 3. Convert HF → Megatron dist (run on each of 4 nodes with NODE_RANK=0..3)
source scripts/models/glm5-744B-A40B.sh
PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/GLM5-744B-A40B \
   --save          $BASE_DIR/GLM5-744B-A40B_torch_dist

# 4. Launch (Python entrypoint)
python scripts/run_glm5_744b_a40b.py
```

## Expected signal

Standard trainer stdout. Frontier-scale-specific:

- Plan for ~1.4 TB host RAM per node when CPU Adam is enabled. If RAM is tight, scale up parallelism until the distributed optimiser holds the state.
- Checkpoint writes at 744 B can saturate most filesystems — raise `--save-interval` if your FS is the bottleneck.

---

## Deep dive

### Launch scripts

| Script | Notes |
|---|---|
| `scripts/run_glm5_744b_a40b.py` | Python launcher |

### Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | Nodes |
|---|---|---|---|---|---|
| 8 | 32 | 4 | 4 | 16 384 | 16 |

### Unified FP8 + R3

Production runs of GLM5 enable:

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

See [FP8 & Low Precision](../../advanced/fp8-low-precision.md) and [Miles Router (R3)](../../advanced/miles-router.md) for the end-to-end configuration.

### Smoke tests

Before committing to a 16-node run, use the shortened configs to iterate:

```bash
# 4-layer smoke test
source scripts/models/glm5-744B-A40B_4layer.sh

# 20-layer smoke test
source scripts/models/glm5-744B-A40B_20layer.sh
```
