---
title: GLM5
description: Launch recipe for GLM5 744B-A40B.
---

# GLM5 744B-A40B

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

Full config in `scripts/models/glm5-744B-A40B.sh`. Short test variants
(`glm5-744B-A40B_4layer.sh`, `glm5-744B-A40B_20layer.sh`) are available for quick
smoke tests.

## Launch scripts

| Script | Notes |
|---|---|
| `scripts/run_glm5_744b_a40b.py` | Python launcher |

## Convert weights (4 nodes)

```bash
hf download zai-org/GLM5-744B-A40B --local-dir $BASE_DIR/GLM5-744B-A40B

cd /root/miles
source scripts/models/glm5-744B-A40B.sh

PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/GLM5-744B-A40B \
   --save          $BASE_DIR/GLM5-744B-A40B_torch_dist
```

## Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | Nodes |
|---|---|---|---|---|---|
| 8 | 32 | 4 | 4 | 16 384 | 16 |

Plan for ~1.4 TB host RAM per node with CPU Adam enabled. Otherwise scale up parallelism
until the distributed optimiser can hold the state.

## Unified FP8 + R3

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

## Launch

```bash
python scripts/run_glm5_744b_a40b.py
```
