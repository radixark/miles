---
title: GLM5
description: Launch recipe for GLM-5 — Python launcher with full / 4-layer / 20-layer variants and 1 / 6 / 16+ node configs.
---

# GLM5

GLM5 is Zhipu's frontier-scale MoE: 744 B total parameters, 40 B active per token, 256 routed experts. Miles ships a 16-node launcher plus 4-layer / 20-layer smoke-test configs for iterating on code changes before paying the 16-node bill. `scripts/run_glm5_744b_a40b.py` is the launcher, which drives the full GLM-5 model plus two pruned smoke-test variants.

## Architecture

| Property | Value |
|---|---|
| Layers | 3 dense + 75 MoE = 78 |
| Routed experts | 256, top-8 |
| Shared experts | 1 |
| Hidden / FFN | 6144 / 12288 (dense) · 2048 (MoE FFN per expert) |
| Heads | 64 |
| Attention | MLA (`--q-lora-rank 2048`, `--kv-lora-rank 512`, `--qk-head-dim 192`, `--v-head-dim 256`, `--qk-pos-emb-head-dim 64`) |
| Vocab | 154880 |
| RoPE base | 1000000 |
| Spec | `--spec miles_plugins.models.glm5.glm5 get_glm5_spec` |
| Router | sigmoid score, pre-softmax, expert bias, `seq_aux_loss`, topk-scaling 2.5 |

Pruned model configs:

- `scripts/models/glm5-744B-A40B_4layer.sh` — single-node smoke test
- `scripts/models/glm5-744B-A40B_20layer.sh` — multi-node smoke test

## Variants (per the launcher's `__post_init__`)

| `--model-name` | HF org | Megatron model type |
|---|---|---|
| `GLM-5` | `zai-org` | `glm5-744B-A40B` |
| `GLM-5_4layer` | `Pinaster` | `glm5-744B-A40B_4layer` |
| `GLM-5_20layer` | `Pinaster` | `glm5-744B-A40B_20layer` |



## Hardware

The launcher's docstring says it's tested on **H200 / B200 / GB300**. The dataclass restricts `--hardware` to `{H200, B200, GB300}`.

## Subcommands

`scripts/run_glm5_744b_a40b.py` is a Typer app with four subcommands (verbatim from the file):

```bash
# Full pipeline: download, convert, copy, train
python scripts/run_glm5_744b_a40b.py full-train --model-name <name> --num-nodes <N>

# Just download model + datasets and convert to Megatron
python scripts/run_glm5_744b_a40b.py prepare    --model-name <name> --num-nodes <N>

# Copy converted checkpoint from shared NFS to local disk (run on every node)
python scripts/run_glm5_744b_a40b.py prepare-cp --model-name <name> --num-nodes <N>

# Train only (assumes prepare/prepare-cp done)
python scripts/run_glm5_744b_a40b.py train      --model-name <name> --num-nodes <N>
```

## Quick start (single-node smoke test)

```bash
ray stop --force && pkill -9 -f sglang || true && sleep 3
ray start --head --port=6378 --dashboard-port=8266

python scripts/run_glm5_744b_a40b.py full-train --model-name GLM-5_4layer --num-nodes 1
```

(For the 4-layer single-node case, `__post_init__` forces `enable_pd=False` and `mode="debug_minimal"`.)

## Quick start (multi-node 20-layer test)

Run from the head node after starting Ray on every node:

```bash
python scripts/run_glm5_744b_a40b.py prepare    --model-name GLM-5_20layer --num-nodes 6
python scripts/run_glm5_744b_a40b.py prepare-cp --model-name GLM-5_20layer --num-nodes 6   # on every node
python scripts/run_glm5_744b_a40b.py train      --model-name GLM-5_20layer --num-nodes 6
```

## Quick start (full GLM-5, 16+ nodes)

```bash
python scripts/run_glm5_744b_a40b.py full-train --model-name GLM-5 --num-nodes 16
```

The launcher patches `config.json` to set `model_type=deepseek_v32` (`_process_glm_checkpoint`) before conversion — GLM-5 is loaded through the DeepseekV32 architecture path.

## Parallelism (verbatim from `_execute_train`)

| `--num-nodes` | TP | PP | CP | EP | expert-TP | `decoder-last-pipeline-num-layers` | `max_tokens_per_gpu` |
|---|---|---|---|---|---|---|---|
| 1 (`GLM-5_4layer`) | 4 | 1 | 1 | 8 (= `num_gpus_per_node`) | 1 | — | 2048 |
| 6 (`GLM-5_20layer`) | 4 | 3 | 1 | 16 | 1 | 6 | 2048 |
| ≥16 (`GLM-5`) | 4 | 4 | 2 | 32 | 1 | 18 | 16384 |

All cases use `--use-dynamic-batch-size`, `--data-pad-size-multiplier 4096`, `--log-probs-chunk-size 1024`, `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`.

## Optional features

The launcher exposes these as flags:

- `--fp8-rollout` — runs `tools/convert_hf_to_fp8.py --strategy block --block-size 128 128` and feeds the FP8 directory to SGLang (Megatron stays BF16)
- `--enable-mtp` — adds SGLang EAGLE speculative decoding (`--sglang-speculative-{algorithm,num-steps,eagle-topk,num-draft-tokens}`)
- `--enable-pd` (default `True` for ≥1 node) — enables prefill/decode disaggregation; with PD the launcher uses larger SGLang world sizes (16 for `<16` nodes, 64 for `≥16` nodes)
- `--enable-optimizer-offload` — adds `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`
- `--use-deepep` (default `True`) — enables Megatron-side DeepEP (`--moe-enable-deepep --moe-token-dispatcher-type flex`); falls back to `alltoall`. Forced off on GB300.

## SGLang (always-on flags)

```bash
--sglang-mem-fraction-static 0.70
--sglang-enable-dp-attention
--sglang-ep-size <world_size>
--sglang-dp-size <world_size>
--sglang-moe-dense-tp-size 1
--sglang-enable-dp-lm-head

# DSA / NSA attention
--sglang-page-size 64
--sglang-nsa-decode-backend flashmla_sparse
--sglang-nsa-prefill-backend flashmla_sparse
--sglang-attention-backend nsa

--sglang-max-running-requests 512
--sglang-watchdog-timeout 3600
```

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`. R3 (`--use-miles-router` etc.) is **not** enabled by default.

## Pairs well with

- [PD Disaggregation](../../advanced/pd-disaggregation.md) — on by default for `num_nodes ≥ 1`.
- [FP8 & Low Precision](../../advanced/fp8-low-precision.md) — opt-in via `--fp8-rollout`.
- [Speculative Decoding](../../advanced/speculative-decoding.md) — opt-in via `--enable-mtp`.
