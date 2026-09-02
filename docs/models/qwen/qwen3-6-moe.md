---
title: Qwen3.6 MoE
description: Launch recipe for Qwen3.6-35B-A3B with MTP training and EAGLE speculative rollout.
---
## 1. Model Introduction

[Qwen3.6-35B-A3B](https://github.com/QwenLM/Qwen3) is the sparse MoE branch of
Alibaba's Qwen3.6 line — 35 B total / 3 B active parameters on a Gated Delta
Networks backbone. Like the dense Qwen3.6-27B, it's tuned for agentic-coding
workflows and long-session reasoning, with native hybrid thinking mode,
built-in tool calling, and multimodal text / image / video input. Context
reaches 262 K and extends past 1 M; weights are Apache 2.0 in BF16 and FP8.
Qwen3.6 also ships native Multi-Token Prediction for speculative decoding,
which this recipe trains and serves via EAGLE.

In miles, Qwen3.6-35B-A3B reuses the Qwen3.5 spec
(`miles_plugins.models.qwen3_5.get_qwen3_5_spec`) and bakes in MTP training
plus a shared-expert gate.

**Key highlights:**

- **Sparse MoE on a GDN backbone**: 256 experts, top-8 routing, 3 B active / 35 B total.
- **Attention-output gate**: shared with the Qwen3.5 / 3.6 dense series.
- **Shared expert + gate**: `--moe-shared-expert-intermediate-size 512 --moe-shared-expert-gate`.
- **Multi-Token Prediction (MTP)**: `--mtp-num-layers 1`; trained alongside the policy and served via EAGLE at rollout.
- **Dispatcher**: `--moe-token-dispatcher-type alltoall` for HF→Megatron conversion; runtime uses `flex` (set in the launcher).
- **Long context**: 262 K tokens, extensible past 1 M.
- **Single-node footprint**: full recipe fits on 1 × 8 GPU (H200).

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Qwen3.6-35B-A3B | 3 B / 35 B | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
hf download Qwen/Qwen3.6-35B-A3B --local-dir /root/models/Qwen3.6-35B-A3B
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3.6-35B-A3B)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/Qwen3.6-35B-A3B \
   --save          /root/models/Qwen3.6-35B-A3B_torch_dist \
   --mtp-num-layers 1
```

`--mtp-num-layers 1` during conversion preserves the MTP layer so it survives into Megatron format.

## 4. Launch

### 4.1 Supervised fine-tuning

The shared Qwen SFT launcher supports Qwen3.6-35B-A3B on one 8 × H200 node.
The dataset can be JSONL or Parquet. Each row's `messages` field must be a
non-empty list of `{role, content}` objects and must include at least one
assistant turn. The launcher renders those messages with the model tokenizer
and trains only the assistant-token positions.

Validate every row with the same tokenizer and loss-mask code used by training:
JSONL is streamed row by row so tool definitions may use heterogeneous nested
schemas. Set `--tools-key` when the conversations include tool definitions.

```bash
python scripts/tools/validate_sft_dataset.py \
   --dataset /root/datasets/train.parquet \
   --model /root/models/Qwen3.6-35B-A3B \
   --tools-key tools \
   --max-seq-len 65536
```

To create a training artifact while preserving every field of accepted rows,
use the companion filter. It writes a content-free rejection audit with the
source row index and exact reason. Any rejection other than an overlong row or
a conversation without an assistant target makes the command fail.

```bash
python scripts/tools/filter_sft_dataset.py \
   --dataset /root/datasets/train.jsonl \
   --output-dataset /root/datasets/train.filtered.jsonl \
   --reject-report /root/datasets/train.rejected.jsonl \
   --summary-path /root/datasets/train.filtered.summary.json \
   --model /root/models/Qwen3.6-35B-A3B \
   --tools-key tools \
   --max-seq-len 262144
```

```bash
cd /root/miles
python scripts/run_qwen3_sft.py \
   --model-name Qwen3.6-35B-A3B \
   --prompt-data /root/datasets/train.parquet \
   --tools-key tools \
   --metadata-key metadata \
   --run-id YYMMDD-8hex \
   --checkpoint-dir /path/to/persistent-storage/qwen36-sft/YYMMDD-8hex/checkpoints \
   --trace-dir /path/to/local-scratch/YYMMDD-8hex/traces
```

Keep checkpoints on storage that survives machine or pod replacement. Details
dumps and traces can use high-throughput node-local scratch storage.

The Qwen3.6 recipe uses `TP=1, EP=8, CP=1, PP=1, expert-TP=1`, Qwen3 loss
masking, CPU-offloaded precision-aware Adam, full activation recomputation,
MTP training, and an 8192-token dynamic budget per GPU. It also enables the
Miles dashboard, Prometheus forwarding, rollout entropy, training entropy, and
the details dump. Dataset-dependent choices such as sequence-length policy,
batch sizes, epoch count, learning rate, and save interval should be selected
after tokenizing the actual data.

For very long sequences, override tensor parallelism and the dynamic token
budget explicitly. Qwen3.6's gated-delta layers do not support context
parallelism in the current Megatron backend, so the launcher rejects `CP>1`.
On one 8-GPU node, `--tensor-model-parallel-size 8 --context-parallel-size 1`
keeps all eight GPUs in one model replica while retaining `EP=8`.

The launcher accumulates and reduces gradients in FP32 by default. For
memory-constrained Muon runs, `--grad-reduce-in-bf16` switches only the gradient
buffers and their reduction to BF16. It does not change logits or loss
precision. This saves device memory at the cost of lower-precision gradient
accumulation and communication.

This is pure SFT: no SGLang rollout engine is started.

### 4.2 RL + MTP quick start

The launcher is a parametrized Typer script (8 × H200) that exercises arbitrary
(TP, EP, CP, PP, ETP) cells:

```bash
cd /root/miles
python scripts/run_qwen3_6_35b_a3b_mtp.py \
   --tp 1 --ep 8 --cp 1 --pp 1 --etp 1 \
   --num-rollout 10
```

Default knobs in the launcher: `--mode debug_minimal`, 8 GPUs, `max_tokens_per_gpu=8192`,
`rollout_batch_size=8`, `n_samples_per_prompt=2`, `global_batch_size=16`,
`rollout_max_response_len=1024`. Override via flags for longer runs.

## 5. Recipe Configuration

### 5.1 Parallelism

The default cell is `TP=1 EP=8 CP=1 PP=1 ETP=1`. Sequence parallelism is on; activation
checkpointing defaults on (`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`)
and can be turned off with `--no-recompute`.

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8 | 1 | 8192 | 8 (1 × 8) |

### 5.2 Algorithm

GRPO with low-variance KL plus MTP training:

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

MTP_ARGS=(
   --enable-mtp-training
   --mtp-num-layers 1
   --mtp-loss-scaling-factor 0.2
)
```

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128

   # MTP speculative decoding
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

   --sglang-max-running-requests 256
   --sglang-mamba-scheduler-strategy extra_buffer
)
```

### 5.4 Optimizer

CPU Adam is enabled (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`).

### 5.5 Notable quirks

From `scripts/models/qwen3.6-35B-A3B.py` and `scripts/run_qwen3_6_35b_a3b_mtp.py`:

- `--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec` — Qwen3.6 reuses the Qwen3.5 spec.
- 256 experts, `--moe-router-topk 8`, `--moe-router-score-function softmax`.
- `--moe-shared-expert-gate` and `--moe-shared-expert-intermediate-size 512`.
- Megatron-side dispatcher overridden to `--moe-token-dispatcher-type flex` at runtime; conversion uses `alltoall`.
- `--moe-grouped-gemm`, `--moe-token-drop-policy probs`, `--moe-router-dtype fp32`, `--moe-permute-fusion`, `--moe-aux-loss-coeff 0`.
- `--attention-output-gate`, `--rotary-base 10000000`, `--rotary-percent 0.25`, `--vocab-size 248320`.

See [Backends Beyond Megatron](/advanced/architecture-support) for FP32 parameter handling and how miles wires the spec.

## 6. Pairs Well With

- [Speculative Decoding](/advanced/speculative-decoding)
- [Backends Beyond Megatron](/advanced/architecture-support)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
