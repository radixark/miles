---
title: Qwen3.5 MoE
description: Launch recipe for Qwen3.5-35B-A3B with MTP training and EAGLE speculative rollout.
---

# Qwen3.5 MoE

`Qwen3.5-35B-A3B` (3 B active / 35 B total). The `mtp` script trains the MTP head and uses EAGLE speculative decoding at rollout.

## Variant

| Model | Active / Total | Local checkpoint path | Model config |
|---|---|---|---|
| Qwen3.5-35B-A3B | 3 B / 35 B | `/root/Qwen3.5-35B-A3B` | `scripts/models/qwen3.5-35B-A3B.sh` |

## Paths the launcher expects

From `run-qwen3.5-35B-A3B-mtp.sh:29-60`:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3.5-35B-A3B
   --ref-load /root/Qwen3.5-35B-A3B_torch_dist
   --load /root/Qwen3.5-35B-A3B_miles/
   --save /root/Qwen3.5-35B-A3B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl ... )
EVAL_ARGS=(   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl ... )
```

## Quick start

```bash
cd /root/miles

# 1. Stage datasets at the paths the script expects
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
# Place the model checkpoint at /root/Qwen3.5-35B-A3B

# 2. Convert HF → Megatron torch_dist  (the launcher will read _torch_dist as --ref-load)
source scripts/models/qwen3.5-35B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3.5-35B-A3B \
   --save          /root/Qwen3.5-35B-A3B_torch_dist \
   --mtp-num-layers 1

# 3. Launch GRPO + MTP
bash scripts/run-qwen3.5-35B-A3B-mtp.sh
```

`--mtp-num-layers 1` during conversion preserves the MTP layer so it survives into Megatron format. The training script then enables MTP training via `MTP_ARGS` (below).

## Launch script

| Script | GPUs | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` |
|---|---|---|---|---|---|---|---|
| `scripts/run-qwen3.5-35B-A3B-mtp.sh` | 8 | 1 | 1 | 1 | 8 | 1 | 8192 |

## MTP training

```bash
MTP_ARGS=(
   --enable-mtp-training
   --mtp-num-layers 1
   --mtp-loss-scaling-factor 0.2
)
```

## SGLang flags

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   # mtp speculative decoding
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

   --sglang-max-running-requests 512
)
```

CPU Adam is enabled (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`). The Megatron side uses `--moe-token-dispatcher-type flex` (DeepEP isn't enabled here, unlike Qwen3-Next).

## Architecture notes

The model config (`scripts/models/qwen3.5-35B-A3B.sh`) reuses the Qwen3.5 spec: `--attention-output-gate`, `--rotary-base 10000000`, `--rotary-percent 0.25`, `A_log` kept in FP32 via the bridge. See [Backends Beyond Megatron](../../advanced/architecture-support.md).
