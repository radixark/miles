---
title: GLM4.7 Flash
description: Launch recipe for GLM4.7 Flash — a compact MoE with 64 routed experts, ideal for single-node routing experiments.
---

# GLM4.7 Flash

GLM4.7 Flash is the compact end of Zhipu's MoE line: 64 routed experts, top-4 activation, one shared expert. It fits on a single 8-GPU node and is a useful testbed for routing / R3 experiments without paying the 355 B price tag of GLM4.5.

## Architecture

| Property | Value |
|---|---|
| Routed experts | 64 |
| Active experts per token | 4 |
| Shared experts | 1 |
| MoE layers | 46 (1 dense head) |
| Hidden size | 2048 |
| Heads | 20 |

Full config lives in `scripts/models/glm4.7-flash.sh`.

## Quick start (8× H100)

```bash
cd /root/miles

# 0. Update transformers — GLM4.7 Flash needs this exact commit
pip install git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa

# 1. Download weights
hf download zai-org/GLM-4.7-Flash --local-dir /root/GLM-4.7-Flash

# 2. Convert HF → Megatron dist checkpoint
source scripts/models/glm4.7-flash.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/GLM-4.7-Flash \
   --save          /root/GLM-4.7-Flash_torch_dist

# 3. Launch GRPO
bash scripts/run-glm4.7-flash.sh
```

## Expected signal

Standard `loss=… reward=…` trainer stdout. MoE-specific: `router/replay_hit_rate` hits ≈ 1.0 quickly when R3 is enabled. GLM4.7 Flash is especially sensitive to router drift — if replay hit-rate does not climb, check that `--use-rollout-routing-replay` and `--sglang-use-miles-router` are both on.

---

## Deep dive

### Launch scripts

| Script | Entry point |
|---|---|
| `scripts/run-glm4.7-flash.sh` | Bash launcher |
| `scripts/run_glm47_flash.py` | Python launcher |

### Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 4 | 8 | 1 | 2 | 8192 | 8 |

### R3 + unified FP8 (recommended)

See [Miles Router (R3)](../../advanced/miles-router.md) and [FP8 & Low Precision](../../advanced/fp8-low-precision.md) for the production configuration. On a compact MoE like GLM4.7 Flash, enabling both is almost always a win: R3 keeps routing stable and FP8 roughly doubles rollout throughput.
