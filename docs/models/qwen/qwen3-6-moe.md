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

### 4.1 Quick start

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

### 4.2 Long-context supervised fine-tuning

The shared Qwen SFT launcher also supports this model on eight GPUs, with
TP=2, CP=4, EP=8, and FP32 gradient accumulation. Supply JSONL rows with a
`messages` field; the Qwen3 mask trains assistant turns, including reasoning,
and masks user/system/tool observations. No inference engines are started.

```bash
python scripts/run_qwen3_sft.py \
   --model-name Qwen3.6-35B-A3B \
   --prompt-data /data/sft_train.jsonl \
   --output-dir /scratch/sft-run \
   --num-epoch 2 --rollout-batch-size 4 --global-batch-size 4 \
   --learning-rate 1e-5 --min-learning-rate 1e-5 \
   --checkpointed-output-projection --log-probs-chunk-size 256 \
   --save-interval 560 \
   --extra-args '--seq-length 262144 --rollout-max-context-len 262144 --lr-warmup-fraction 0'
```

For 1,120 rows, this is 560 optimizer updates. Checkpoint saving is also
triggered at epoch boundaries and the final update. Use an output directory
with sufficient space; optimizer checkpoints are considerably larger than
the inference weights. The checkpointed projection recomputes bounded logits
chunks during backward. Cross-entropy reductions remain FP32, and parameter
gradient precision is unchanged. This option requires a Megatron GPT model
with the output-processor interface and `LinearCrossEntropyModule` support.

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

## 6. SFT data preflight

Before training a new JSONL corpus, compare the SFT tokenizer and assistant-only loss
mask against the complete chat template with `preserve_thinking=True`:

```bash
python tools/check_qwen_sft_masks.py \
    --data /data/sft.jsonl \
    --tokenizer /models/Qwen3.6-35B-A3B \
    --output /data/mask-audit.json \
    --workers 8
```

The audit checks every row, includes top-level tool definitions, and rejects token
mismatches, empty loss targets, and sequences beyond 262,144 tokens. It writes numeric
diagnostics without conversation text. For corpora with a top-level `tools` field,
also pass `--tool-key tools` to training so the data loader forwards those definitions.

For a pure SFT run without the auxiliary MTP objective, pass `--no-enable-mtp` to
`scripts/run_qwen3_sft.py --model-name Qwen3.6-35B-A3B`. This omits
`--enable-mtp-training` and overrides the model definition with `--mtp-num-layers 0`,
so no MTP block is constructed. Setting only the MTP loss weight to zero is not
equivalent. The main assistant-token mask and optimizer settings are unchanged.

## 7. Pairs Well With

- [Speculative Decoding](/advanced/speculative-decoding)
- [Backends Beyond Megatron](/advanced/architecture-support)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
