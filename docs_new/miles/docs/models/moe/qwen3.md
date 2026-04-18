---
title: Qwen3 MoE
description: Launch recipes for Qwen3 30B-A3B and Qwen3 235B-A22B.
---

# Qwen3 MoE

## Variants

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Qwen3-30B-A3B | 3 B / 30 B | `Qwen/Qwen3-30B-A3B` | `scripts/models/qwen3-30B-A3B.sh` |
| Qwen3-235B-A22B | 22 B / 235 B | `Qwen/Qwen3-235B-A22B` | `scripts/models/qwen3-235B-A22B.sh` |

FP8 variants (`Qwen/Qwen3-30B-A3B-FP8`, etc.) are drop-in replacements for the HF
checkpoint path — see [FP8 & Low Precision](../../advanced/fp8-low-precision.md).

## Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-qwen3-30B-A3B.sh` | 8× H100 | Canonical recipe, single node |
| `scripts/run-qwen3-235B-A22B.sh` | 32× H100 | Multi-node |
| `scripts/run-qwen3-235B-A22B-sft.sh` | 32× H100 | SFT variant |

## Convert weights

```bash
cd /root/miles
source scripts/models/qwen3-30B-A3B.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/Qwen3-30B-A3B \
   --save          /data/Qwen3-30B-A3B_torch_dist
```

## Parallelism

| Size | TP | EP | PP | CP | `max_tokens_per_gpu` | Nodes |
|---|---|---|---|---|---|---|
| 30B-A3B | 4 | 8 | 1 | 1 | 4608 | 1 |
| 235B-A22B | 8 | 16 | 2 | 2 | 8192 | 4 |

On a single node, Qwen3-30B-A3B needs CPU Adam to fit the optimiser state:

```bash
OPTIMIZER_ARGS+=(
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)
```

Drop those flags once you go multi-node — the distributed optimiser handles the memory
pressure.

## SGLang MoE flags

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-enable-ep-moe
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)
```

For multi-node (e.g. 24 GPUs), add redundant experts:

```bash
SGLANG_ARGS+=(
   --sglang-enable-dp-attention
   --sglang-dp-size 3
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-ep-num-redundant-experts 16
)
```

## R3

Enable Rollout Routing Replay for stable MoE training:

```bash
GRPO_ARGS+=(
   --use-miles-router
   --use-rollout-routing-replay
   --use-tis
)
SGLANG_ARGS+=( --sglang-use-miles-router )
```

See [Miles Router (R3)](../../advanced/miles-router.md) for the rationale.

## Launch

```bash
bash scripts/run-qwen3-30B-A3B.sh
```

## Tuning

| Symptom | Fix |
|---|---|
| OOM in optimiser | Add CPU Adam flags (single-node only) |
| OOM in rollout | Drop `--sglang-mem-fraction-static` to 0.6 |
| One expert dominating routes | Confirm R3 is active (`router/replay_hit_rate ≈ 1.0`) |
| Train slower than rollout | Disable CPU offload (multi-node), raise `max-tokens-per-gpu` |
