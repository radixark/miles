---
title: GLM-5.2
description: Launch recipe for GLM-5.2 (744 B / 40 B active) — FP8 KV cache, TIS, 16+ node config.
---
## 1. Model Introduction

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) is the successor to GLM-5 / GLM-5.1 in Zhipu AI's GLM series. It keeps the same 744 B-parameter (40 B active) `glm_moe_dsa` architecture — MoE plus DeepSeek Sparse Attention (DSA) with cross-layer index sharing — and differs from GLM-5 in the checkpoint it loads, the Megatron model args (`--rotary-base 8000000`), and the rollout recipe (FP8 KV cache, `flashmla_kv` decode, optional EAGLE speculative decoding).

**Key highlights:**

- **Sparse MoE at frontier scale**: 744 B total / 40 B active per token, 256 routed experts top-8 + 1 shared, 3 dense + 75 MoE layers.
- **MLA + DSA with cross-layer index sharing**: only the *computing* layers (1, 2, 3, 7, 11, …, 75 in Megatron 1-indexing; `index_topk_freq=4`) carry indexer weights and compute the sparse top-k — the remaining layers reuse the most recent computing layer's indices. This constrains the pipeline split: every PP stage must start on a computing layer.
- **FP8 KV cache rollout**: `fp8_e4m3` KV cache with `flashmla_kv` decode and `flashmla_sparse` prefill.
- **Truncated importance sampling**: the GLM-5.2 recipe enables TIS by default (`--use-tis`).

## 2. Supported Variants

| Model | Active / Total | HF ID | Scale |
|---|---|---|---|
| GLM-5.2 | 40 B / 744 B (78 layers) | [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | ≥ 16 nodes |
| GLM-5.2_5layer | 5-layer pruned | Pinaster/GLM-5.2_5layer | 1 node (smoke test) |

A LoRA variant of the recipe ships as `scripts/run_glm5_2_744b_a40b_lora.py` (see [LoRA](/advanced/lora)).

## 3. Environment Setup

Use the `radixark/miles:dev` docker image.

### 3.1 Download model + datasets

The Python launcher's `prepare` subcommand handles download + dataset staging (dapo-math-17k):

```bash
python scripts/run_glm5_2_744b_a40b.py prepare --model-name GLM-5.2 --num-nodes 32
```

### 3.2 HF → Megatron `torch_dist` conversion

Also handled by `prepare`. Before conversion the launcher validates, via `_validate_glm_checkpoint`, that the checkpoint uses the native GLM-5.2 config (`model_type=glm_moe_dsa`, `architectures=[GlmMoeDsaForCausalLM]`, `num_hidden_layers=78`, no `auto_map`) and fails fast if it does not, then converts it to the `glm5.2-744B-A40B` Megatron model type. The full model converts with PP = 4 (18/20 first/last layer split); the pruned model converts on a single GPU, because DSA's cross-layer index sharing forbids a pipeline stage that starts on a skip layer. Run `prepare-cp` afterwards on every node to copy the converted checkpoint from shared NFS to local disk.

## 4. Launch

### 4.1 Quick start

Single-node smoke test with the 5-layer pruned model:

```bash
python scripts/run_glm5_2_744b_a40b.py full-train --model-name GLM-5.2_5layer --num-nodes 1
```

Full model (≥ 16 nodes):

```bash
python scripts/run_glm5_2_744b_a40b.py full-train --model-name GLM-5.2 --num-nodes 32
```

The Typer app exposes four subcommands:

```bash
python scripts/run_glm5_2_744b_a40b.py full-train --model-name GLM-5.2 --num-nodes <N>

# Just download model + datasets and convert to Megatron
python scripts/run_glm5_2_744b_a40b.py prepare    --model-name GLM-5.2 --num-nodes <N>

# Copy converted checkpoint from shared NFS to local disk (run on every node)
python scripts/run_glm5_2_744b_a40b.py prepare-cp --model-name GLM-5.2 --num-nodes <N>

# Train only (assumes prepare/prepare-cp done)
python scripts/run_glm5_2_744b_a40b.py train      --model-name GLM-5.2 --num-nodes <N>
```

The recipe is tested on **H200 / B200 / GB300**; the `--hardware` flag accepts exactly these three values.

### 4.2 Agentic RL: terminal-bench-2 in Daytona sandboxes (experimental)

Beyond the math recipe above, [`examples/experimental/openenv/glm52_tbench2/`](https://github.com/radixark/miles/tree/main/examples/experimental/openenv/glm52_tbench2) trains GLM-5.2 with fully-async agentic RL on terminal-bench-2: 16 GB300 nodes (4 GPUs each) split into 8 training nodes (TP2 / CP4 / PP4 / EP8, optimizer state streamed to node-local disk) and 8 inference nodes (one 4-GPU dp-attention FP8 SGLang engine per node). Every episode is a multi-turn terminal agent solving one terminal-bench-2 task inside its own Daytona cloud sandbox built from that task's official image; scoring is the task's canonical `tests/test.sh`.

```bash
python3 examples/experimental/openenv/glm52_tbench2/run_glm5_2_744b_a40b_daytona.py train --num-nodes 16
```

The recipe's defaults are the reference configuration (100 rollout steps in ~21 h, ~6.5 min/step including evals). See the example's README for the container, Megatron, OpenEnv, and Daytona prerequisites.

## 5. Recipe Configuration

### 5.1 Parallelism

`_execute_train` picks one of three branches:

| Branch | TP | PP | CP | EP | expert-TP | first/last PP layers | `max_tokens_per_gpu` |
|---|---|---|---|---|---|---|---|
| 1 node (5-layer) | 4 | 1 | 1 | = GPUs per node | 1 | — | 2048 |
| ≥ 16 nodes, 4 GPUs/node (GB300) | 8 | 4 | 1 | 16 | 1 | 18 / 20 | 8192 |
| ≥ 16 nodes, 8 GPUs/node | 4 | 8 | 8 | 32 | 1 | 14 / 16 | 8192 |

The uneven first/last pipeline splits are dictated by DSA: every stage must start on a computing layer (e.g. the 14/16 split lands stage starts on layers 1, 15, 23, 31, 39, 47, 55, 63 — all computing).

Plus `--use-dynamic-batch-size`, `--data-pad-size-multiplier 1024`, `--log-probs-chunk-size 16384`, `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`, and `--allgather-cp` (DSA + context parallel uses the sequential allgather-CP layout).

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`, plus truncated importance sampling — `--use-tis --tis-clip-low 0.5 --tis-clip 2.0` — which the GLM-5 recipe does not enable. R3 (`--use-rollout-routing-replay`) is **not** enabled by default.

### 5.3 Rollout & SGLang

Always-on flags:

```bash
--sglang-mem-fraction-static 0.85   # 0.70 on the 1-node smoke test
--sglang-ep-size <world_size>
--sglang-router-policy consistent_hashing

# DSA / NSA attention with FP8 KV cache
--sglang-kv-cache-dtype fp8_e4m3
--sglang-nsa-decode-backend flashmla_kv
--sglang-nsa-prefill-backend flashmla_sparse
--sglang-attention-backend nsa
--sglang-page-size 64

--sglang-max-running-requests 512   # 256 with --sglang-config balanced
--sglang-watchdog-timeout 3600
```

With PD disaggregation (the multi-node default) the launcher also adds `--sglang-enable-dp-attention --sglang-dp-size <world_size> --sglang-moe-dense-tp-size 1 --sglang-enable-dp-lm-head` and uses SGLang world size 16 (`< 16` nodes) or 64 (`≥ 16` nodes).

Training runs with these env vars: `SGLANG_NSA_FORCE_MLA=1`, `INDEXER_ROPE_NEOX_STYLE=0`, `NVSHMEM_DISABLE_NCCL=1`.

### 5.4 Optimizer

Adam, `--lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98`. `--enable-optimizer-offload` adds `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer` (opt-in).

### 5.5 Notable quirks

The launcher exposes these as flags:

- `--fp8-rollout` — runs `tools/convert_hf_to_fp8.py --strategy block --block-size 128 128` and feeds the FP8 directory to SGLang (Megatron stays BF16). Combined with `--use-deepep` it also switches SGLang's MoE all-to-all to DeepEP (`--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto`).
- `--enable-mtp` — adds SGLang EAGLE speculative decoding with `--sglang-speculative-draft-attention-backend nsa`; `low-latency` drafts deeper (num-steps 5, draft-tokens 6) than `balanced` (1, 2). Full model only — the MTP layer is pruned away in the 5-layer variant.
- `--enable-pd` (default `True`, forced off on 1 node) — enables prefill/decode disaggregation.
- `--sglang-config {low-latency, balanced}` — `low-latency` (default) runs TP-8 engines with PD; `balanced` is the GLM-5.2 cookbook serving shape: one 4-GPU engine per node with dp-attention + DeepEP. `balanced` is incompatible with PD and requires GPUs per node divisible by 4.
- `--use-deepep` (default `True`) — enables Megatron-side DeepEP (`--moe-enable-deepep --moe-token-dispatcher-type flex`); falls back to `alltoall`. On GB300 you must pass `--no-megatron-use-deepep` (known Megatron DeepEP failure; the launcher asserts).
- On B200/GB300 (without `balanced` or FP8 + DeepEP) the launcher pins `--sglang-moe-runner-backend`: `flashinfer_trtllm_routed` for FP8 rollout, `triton` for BF16.

## 6. Pairs Well With

- [PD Disaggregation](/advanced/pd-disaggregation) — on by default for multi-node runs.
- [Low Precision RL](/advanced/fp8-low-precision) — opt-in via `--fp8-rollout`.
- [Speculative Decoding](/advanced/speculative-decoding) — opt-in via `--enable-mtp`.
- [LoRA](/advanced/lora) — via `scripts/run_glm5_2_744b_a40b_lora.py`.
- [Fully Async Rollout](/examples/fully-async) — the terminal-bench-2 agentic example (§4.2) runs fully async.
