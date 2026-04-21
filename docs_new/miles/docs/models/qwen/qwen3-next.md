---
title: Qwen3-Next
description: Launch recipe for Qwen3-Next 80B-A3B (Gated-Delta-Net) on the HuggingFace-wrapped Megatron backend.
---

# Qwen3-Next 80B-A3B

Qwen3-Next swaps classical attention for Gated-Delta-Net (GDN). Miles runs it through the HuggingFace-wrapped Megatron backend, which loads the `Qwen/Qwen3-Next-80B-A3B` HF module as a Megatron stage without re-implementing GDN from scratch.

## Variant

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Qwen3-Next-80B-A3B | 3 B / 80 B | `Qwen/Qwen3-Next-80B-A3B` | `scripts/models/qwen3-next-80B-A3B.sh` |

## Quick start (16× H100, 2 nodes)

```bash
cd /root/miles

# 1. Env — both must be set before launch
export BASE_FOLDER=/shared/checkpoints   # reachable from every node
export MASTER_ADDR=<head node IP>

# 2. Download weights
hf download Qwen/Qwen3-Next-80B-A3B --local-dir $BASE_FOLDER/Qwen3-Next-80B-A3B

# 3. Convert HF → Megatron dist checkpoint
source scripts/models/qwen3-next-80B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_FOLDER/Qwen3-Next-80B-A3B \
   --save          $BASE_FOLDER/Qwen3-Next-80B-A3B_torch_dist

# 4. Launch GRPO
bash scripts/run-qwen3-next-80B-A3B.sh
```

An 8-GPU tight-fit variant (`run-qwen3-next-80B-A3B-8gpus.sh`) is available for single-node smoke tests.

## Expected signal

Standard `loss=… reward=…` trainer stdout. Things specific to Qwen3-Next:

- With the HF-wrapped backend, trainer start-up is slower than native Megatron — the first iteration prints only after HF module construction completes across all ranks.
- Because the backend wraps HF, weight-conversion failures usually surface as "missing key" errors at load time rather than at conversion time.

---

## Deep dive

### Launch scripts

| Script | GPUs | Notes |
|---|---|---|
| `scripts/run-qwen3-next-80B-A3B.sh` | 16× H100 | Canonical, Megatron backend |
| `scripts/run-qwen3-next-80B-A3B-8gpus.sh` | 8× H100 | Tight-fit variant |
| `scripts/run-qwen3-next-80B-A3B-fsdp.sh` | 16× H100 | FSDP backend, skips weight conversion |

### Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 | 8 | 2 | 1 | 4608 | 16 |

With the HF-wrapped backend, **TP inside the Attention module is not supported** (see the [limitations](../../advanced/architecture-support.md#current-limitations)). If you need TP there, fall back to a native Megatron implementation.

### FSDP variant

FSDP skips the weight conversion step entirely:

```bash
bash scripts/run-qwen3-next-80B-A3B-fsdp.sh
```

See [Backends Beyond Megatron](../../advanced/architecture-support.md) for the HF-wrapped backend contract and [Miles Router (R3)](../../advanced/miles-router.md) for MoE routing stability.
