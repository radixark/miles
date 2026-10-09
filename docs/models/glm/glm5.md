---
title: GLM-5 / GLM-5.1
description: Launch recipe for GLM-5 and GLM-5.1 (744 B / 40 B active) — Python launcher, 16+ node config.
---
## 1. Model Introduction

[GLM-5](https://huggingface.co/zai-org/GLM-5) is the most powerful language model in Zhipu AI's GLM series, scaling to 744 B parameters (40 B active) and integrating DeepSeek Sparse Attention (DSA) for long-context efficiency. [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) is the next-generation model for agentic engineering on top of GLM-5, sharing the same model architecture. For the newer GLM-5.2 recipe, see [GLM-5.2](/models/glm/glm5-2).

**Key highlights:**

- **Sparse MoE at frontier scale**: 744 B total / 40 B active per token, 256 routed experts top-8 + 1 shared.
- **MLA + DSA attention**: Multi-head Latent Attention (q-LoRA 2048 / kv-LoRA 512) combined with DeepSeek Sparse Attention to keep KV-cache cost low at long context.
- **Speculative decoding**: EAGLE/MTP rollout supported via `--enable-mtp`.
- **PD disaggregation**: prefill/decode disaggregation enabled by default for ≥1 node.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| GLM-5.1 | 40 B / 744 B | [zai-org/GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) |
| GLM-5 | 40 B / 744 B | [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) |

## 3. Environment Setup

### 3.1 Download model + datasets

The Python launcher's `prepare` subcommand handles download + dataset staging:

```bash
python scripts/run_glm5_744b_a40b.py prepare --model-name GLM-5 --num-nodes 16
```

### 3.2 HF → Megatron `torch_dist` conversion

Also handled by `prepare`. Before conversion the launcher validates, via `_validate_glm_checkpoint`, that the checkpoint uses the native GLM-5 config (`model_type=glm_moe_dsa`, `architectures=[GlmMoeDsaForCausalLM]`) and fails fast if it does not, then converts it to the `glm5-744B-A40B` Megatron model type. Training itself copies the converted checkpoint from shared NFS to each node's local disk before it starts.

### 3.3 DSA training implementation

The raw Megatron path supports `--dsa-impl miles|megatron` for DSA models using the shared GLM-5 / DeepSeek-V3.2 spec. `miles` remains the default. To select Megatron's native DSA without Megatron-Bridge, pass the following to both `tools/convert_hf_to_torch_dist.py` and training:

```bash
--megatron-to-hf-mode raw --dsa-impl megatron --dsa-kernel-backend cudnn
```

Training uses packed `--qkv-format thd` and supports sequence parallelism. Native context parallelism uses zigzag token partitioning within each sequence and `--cp-comm-type allgather` for attention communication. Omitting `--cp-comm-type` selects `allgather` for native DSA; explicitly selecting another communication type is rejected. Miles' `--allgather-cp` instead selects contiguous token partitioning and is rejected for native DSA when CP > 1. GLM-5.2's cross-layer index sharing schedule is preserved. Indexer replay is outside the scope of this backend integration.

The fused cuDNN indexer supports CP1, single-sequence CP batches, and complete multi-sequence CP query partitions. TP-local query slices of multi-sequence CP batches are rejected; select the reference implementation explicitly for that layout. Fused execution uses the configured cuDNN frontend directly and does not silently switch to reference scoring.

`--miles-dsa-topk-backend` selects top-k for both implementations and is accepted by conversion, training, and the standalone `run_megatron` debug worker. This runtime selection does not change the checkpoint layout. The W4A16 test keeps `flashinfer` and `SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=large`, preserving its top-k backend and tie policy when selecting native Megatron DSA.

Native DSA defaults to `--dsa-indexer-loss-coeff 0` and freezes its indexer parameters, because native top-k selection runs without gradients when the auxiliary objective is disabled. This excludes unused parameters from DDP and optimizer weight decay. A positive native loss coefficient keeps the indexer trainable and requires explicitly selecting the reference implementation with `--dsa-kernel-backend none` or `--attention-backend unfused` when using an external top-k backend.

Convert into a separate `torch_dist` directory when changing implementations: the native attention and indexer parameter names differ from the Miles checkpoint layout. Use the same implementation for conversion, checkpoint loading, and training. This selector applies to DSA training; SGLang rollout backend flags remain independent.

With BF16 training, native cuDNN uses BF16 head weights; Miles produces FP32 head weights from BF16 projection operands. [cuDNN Frontend PR #1311](https://github.com/NVIDIA/cudnn-frontend/pull/1311) adds FP32 input support to forward scoring only. This wiring remains BF16; FP32 projection and auxiliary-loss recompute/backward support are separate work. Head-weight precision affects scores independently of the top-k backend and tie policy.

## 4. Launch

### 4.1 Quick start

```bash
python scripts/run_glm5_744b_a40b.py full-train --model-name GLM-5 --num-nodes 16
```

The Typer app exposes three subcommands:

```bash
python scripts/run_glm5_744b_a40b.py full-train --model-name GLM-5 --num-nodes <N>

# Just download model + datasets and convert to Megatron
python scripts/run_glm5_744b_a40b.py prepare    --model-name GLM-5 --num-nodes <N>

# Train only (assumes prepare done); copies the checkpoint to each node's local disk first
python scripts/run_glm5_744b_a40b.py train      --model-name GLM-5 --num-nodes <N>
```

The recipe is tested on **H200 / B200 / GB300**; the `--hardware` flag accepts exactly these three values.

## 5. Recipe Configuration

### 5.1 Parallelism

Verbatim from `_execute_train`, `--num-nodes ≥ 16` branch:

| TP | PP | CP | EP | expert-TP | `decoder-last-pipeline-num-layers` | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|
| 4 | 4 | 2 | 32 | 1 | 18 | 16384 | ≥ 128 (≥ 16 × 8) |

Plus `--use-dynamic-batch-size`, `--data-pad-size-multiplier 4096`, `--log-probs-chunk-size 1024`, `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`. R3 (`--use-rollout-routing-replay`) is **not** enabled by default.

### 5.3 Rollout & SGLang

Always-on flags:

```bash
--sglang-mem-fraction-static 0.70
--sglang-enable-dp-attention
--sglang-ep-size <world_size>
--sglang-dp-size <world_size>
--sglang-moe-dense-tp-size 1
--sglang-enable-dp-lm-head

# DSA / NSA attention
--sglang-page-size 64
--sglang-dsa-decode-backend flashmla_sparse
--sglang-dsa-prefill-backend flashmla_sparse
--sglang-attention-backend nsa

--sglang-max-running-requests 512
--sglang-watchdog-timeout 3600
```

### 5.4 Optimizer

`--enable-optimizer-offload` adds `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer` (opt-in).

### 5.5 Notable quirks

The launcher exposes these as flags:

- `--fp8-rollout` — runs `tools/convert_hf_to_fp8.py --strategy block --block-size 128 128` and feeds the FP8 directory to SGLang (Megatron stays BF16).
- `--enable-mtp` — adds SGLang EAGLE speculative decoding (`--sglang-speculative-{algorithm,num-steps,eagle-topk,num-draft-tokens}`).
- `--enable-pd` (default `True` for ≥1 node) — enables prefill/decode disaggregation; with PD the launcher uses larger SGLang world sizes (16 for `<16` nodes, 64 for `≥16` nodes).
- `--use-deepep` (default `True`) — enables Megatron-side DeepEP (`--moe-enable-deepep --moe-token-dispatcher-type flex`); falls back to `alltoall`. Forced off on GB300.

## 6. Pairs Well With

- [PD Disaggregation](/advanced/pd-disaggregation) — on by default for `num_nodes ≥ 1`.
- [Low Precision RL](/advanced/low-precision) — opt-in via `--fp8-rollout`.
- [Speculative Decoding](/advanced/speculative-decoding) — opt-in via `--enable-mtp`.
