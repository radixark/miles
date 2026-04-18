---
title: DeepSeek R1 / V3
description: Launch recipe for DeepSeek-R1 and DeepSeek-V3 at 671 B total / 37 B active.
---

# DeepSeek R1 / V3

The largest model family Miles ships. 671 B total parameters, 37 B active per token,
multi-rack training with dynamic sampling and FP8 inference.

## Variants

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| DeepSeek-V3 | 37 B / 671 B | `deepseek-ai/DeepSeek-V3` | `scripts/models/deepseek-v3.sh` |
| DeepSeek-R1 | 37 B / 671 B | `deepseek-ai/DeepSeek-R1` | `scripts/models/deepseek-v3.sh` |

Short test variants (`deepseek-v3-5layer.sh`, `deepseek-v3-20layer.sh`) exist for quick
smoke tests.

## Launch scripts

| Script | Entry point |
|---|---|
| `scripts/run-deepseek-r1.sh` | Bash launcher (16 nodes) |
| `scripts/run_deepseek.py` | Python launcher |

## Configuration summary

| Property | Value |
|---|---|
| Training precision | BF16 |
| Inference precision | FP8 (128×128 block-wise) |
| Max response length | 32 768 tokens |
| SGLang parallelism | EP64, DP-attention, DeepEP |
| Megatron parallelism | TP8 / PP4 / EP32 / CP4 |
| Optimiser | CPU Adam (~1.4 TB host RAM / node) |

## Convert weights

DeepSeek's HF release ships in block-quantised FP8. Convert to BF16 first:

```bash
hf download deepseek-ai/DeepSeek-R1 --local-dir $BASE_DIR/DeepSeek-R1

cd /root/miles
python tools/fp8_cast_bf16.py \
  --input-fp8-hf-path  $BASE_DIR/DeepSeek-R1 \
  --output-bf16-hf-path $BASE_DIR/DeepSeek-R1-bf16
```

Then convert BF16 → Megatron `torch_dist` across 4 nodes:

```bash
source scripts/models/deepseek-v3.sh

PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 8 \
   --expert-tensor-parallel-size 1 \
   --expert-model-parallel-size 4 \
   --decoder-first-pipeline-num-layers 7 \
   --decoder-last-pipeline-num-layers 6 \
   --hf-checkpoint $BASE_DIR/DeepSeek-R1-bf16 \
   --save          $BASE_DIR/DeepSeek-R1_torch_dist
```

## Parallelism

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 4
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)
```

With CP = 4, a single CP group shares a 128 K-token budget.

## SGLang flags

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 64
   --sglang-mem-fraction-static 0.7
   --sglang-enable-ep-moe
   --sglang-enable-dp-attention
   --sglang-enable-deepep
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)
```

## Dynamic sampling (DAPO-style)

```bash
ROLLOUT_ARGS+=(
   --over-sampling-batch-size 192
   --dynamic-sampling-filter-path \
      miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)
```

## R3 + TIS

```bash
GRPO_ARGS+=(
   --use-miles-router
   --use-rollout-routing-replay
   --use-tis
)
SGLANG_ARGS+=( --sglang-use-miles-router )
```

## Launch (16 nodes)

```bash
bash scripts/run-deepseek-r1.sh       # on node 0
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 \
          --node-ip-address ${WORKER_IP} --disable-usage-stats   # on every other node
```

## Tuning

| Symptom | Fix |
|---|---|
| Host OOM | Add nodes. Don't swap. |
| EP imbalance | Add `--sglang-ep-num-redundant-experts 32` |
| Weight sync slow | Check `NCCL_IB_HCA` is pointing at the right NICs |
| First eval timeout | Lower `--n-samples-per-eval-prompt` or response length |

## Pairs well with

- [PD Disaggregation](../../advanced/pd-disaggregation.md)
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md)
- [Fault Tolerance](../../advanced/fault-tolerance.md)
