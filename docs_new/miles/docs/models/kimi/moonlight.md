---
title: Moonlight
description: Launch recipe for Moonlight 16B-A3B — a compact MoE for single-node MoE RL experimentation.
---

# Moonlight 16B-A3B

Moonlight is Moonshot's compact MoE — 16 B total, 3 B active — and a useful single-node test target for MoE RL code changes before scaling to Kimi K2. The whole recipe fits on 8× H100.

## Variant

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Moonlight-16B-A3B | 3 B / 16 B | `moonshotai/Moonlight-16B-A3B` | `scripts/models/moonlight.sh` |

## Quick start (8× H100)

```bash
cd /root/miles

# 1. Download weights
hf download moonshotai/Moonlight-16B-A3B --local-dir /root/Moonlight-16B-A3B

# 2. Convert HF → Megatron torch_dist
source scripts/models/moonlight.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Moonlight-16B-A3B \
   --save          /root/Moonlight-16B-A3B_torch_dist

# 3. Launch GRPO
bash scripts/run-moonlight-16B-A3B.sh
```

## Expected signal

Standard `loss=… reward=…` trainer stdout. On a single node, the optimiser state can be tight — enable CPU Adam if you OOM in the first few steps (see the Deep dive below).

---

## Deep dive

### Launch scripts

| Script | GPUs |
|---|---|
| `scripts/run-moonlight-16B-A3B.sh` | 8× H100 |

### Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 | 8 | 1 | 1 | 4608 | 8 |

### CPU Adam

If single-node memory is tight:

```bash
OPTIMIZER_ARGS+=(
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)
```

### Related

Enable R3 for stability — see [Miles Router (R3)](../../advanced/miles-router.md).
