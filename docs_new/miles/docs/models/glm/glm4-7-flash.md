---
title: GLM4.7 Flash
description: Launch recipes for GLM-4.7-Flash — compact MLA + MoE with R3 enabled by default.
---

# GLM4.7 Flash

GLM-4.7-Flash (`zai-org/GLM-4.7-Flash`) is the compact end of Zhipu's MoE line: 64 routed experts, top-4 routing, MLA attention. Both miles launchers (`scripts/run-glm4.7-flash.sh` and `scripts/run_glm47_flash.py`) target a single 8-GPU node and **enable R3 by default** (`--use-miles-router --use-rollout-routing-replay`).

## Architecture

| Property | Value |
|---|---|
| Layers | 1 dense + 46 MoE = 47 |
| Routed experts | 64, top-4 |
| Shared experts | 1 (`--moe-shared-expert-intermediate-size = 1 × moe_ffn_hidden`) |
| Hidden / FFN | 2048 / 10240 (dense) · 1536 (MoE FFN per expert) |
| Heads | 20 (`--num-attention-heads 20`) |
| Attention | MLA (`--multi-latent-attention`, `--q-lora-rank 768`, `--kv-lora-rank 512`, `--qk-head-dim 192`, `--v-head-dim 256`, `--qk-pos-emb-head-dim 64`) |
| Vocab | 154880 |
| RoPE base | 1000000 |
| MTP | `--mtp-num-layers 1` (baked into the model config) |
| Router | sigmoid score, pre-softmax, expert bias, `seq_aux_loss`, topk-scaling 1.8 |

## Variants & launchers

| Launcher | HF ID | Default hardware | Datasets |
|---|---|---|---|
| `scripts/run-glm4.7-flash.sh` (bash) | `zai-org/GLM-4.7-Flash` | uses `BASE_DIR=/root/shared` (hardcoded L29) | `$BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl`, `$BASE_DIR/aime-2024/aime-2024.jsonl` |
| `scripts/run_glm47_flash.py` (Python, Typer) | `zai-org/GLM-4.7-Flash` | `H200` (only literal allowed in the dataclass) | downloads `zhuzilin/dapo-math-17k`, `zhuzilin/aime-2024` automatically |

## Quick start (bash launcher, 1 node × 8 GPU)

```bash
cd /root/miles
# Stage everything under /root/shared/ (the script's hardcoded BASE_DIR)
hf download zai-org/GLM-4.7-Flash --local-dir /root/shared/GLM-4.7-Flash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/shared/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/shared/aime-2024

source scripts/models/glm4.7-flash.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/shared/GLM-4.7-Flash \
   --save          /root/shared/GLM-4.7-Flash_torch_dist

bash scripts/run-glm4.7-flash.sh
```

## Quick start (Python launcher, H200)

`scripts/run_glm47_flash.py` does the download + checkpoint conversion automatically:

```bash
cd /root/miles
python scripts/run_glm47_flash.py
```

Defaults (see `ScriptArgs`): `model_org=zai-org`, `model_name=GLM-4.7-Flash`, `num_gpus_per_node=8`, `hardware=H200`, `data_dir=/root/datasets`, `model_dir=/root/models`.

## Parallelism

Both launchers ship the same shape:

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8 | 1 | 32768 | 8 (1 × 8) |

`--rollout-num-gpus-per-engine 4` (TP must divide 20 attention heads, so TP=4). The bash launcher's `SGLANG_ARGS` keeps `--sglang-enable-dp-attention` / `--sglang-dp-size` commented out; the in-source comment notes that DP-attention requires `tp_size % dp_size == 0`.

## SGLang flags

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7

   # EAGLE speculative decoding (MTP)
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

   # R3 — on by default in this script
   --use-miles-router
   --use-rollout-routing-replay
)
```

CPU Adam is enabled (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`). Megatron-side DeepEP / `flex` dispatcher are commented out by default.

## Pairs well with

- [Miles Router (R3)](../../advanced/miles-router.md) — already on; this is the rationale.
- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
