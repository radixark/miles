---
title: Inkling 2 Mini
description: Full-parameter RL recipe for Inkling 2 Mini, a looped Block Attention Residuals MoE, on the torchtitan backend with colocated SGLang.
---

## 1. Model Introduction

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is the smallest model of Thinking Machines Lab's Inkling 2 family, a mixture-of-experts model with **26B total and 5.4B active parameters**. Like [Inkling 2 Small](/models/thinkingmachines/inkling-2-small), it combines hybrid local and global relative attention with **Block Attention Residuals**, and it adds a **looped core**: a middle span of layers runs several times per token, with a learned gate per pass.

Miles trains it on the [torchtitan backend](/user-guide/training-backend) (`miles/backends/torchtitan_utils/models/inkling2/`): FSDP2 with expert parallelism, colocated SGLang rollout engines. The loop and the attention residuals are plain PyTorch modules, so they need no pipeline-parallel partitioning.

**Key highlights:**

- **Looped core**: layers 6–17 run three passes, each pass mixed back in through a low-rank gate; the trainer unrolls the passes and backpropagates through all of them.
- **Block Attention Residuals**: each layer mixes the completed residual blocks and the current partial block with a learned softmax over depth, instead of adding into one residual stream.
- **Relative attention**: 1,024-token sliding-window and full-attention layers, a learned relative-position bias and a gated output. Packed documents stay isolated in the trainer.
- **Sigmoid-routed MoE**: 128 routed experts at top-8 with soft-capped SwiGLU; no shared experts.
- **BF16 end to end**: the trainer and the rollout engines read the same BF16 checkpoint, dequantized once from the release's NVFP4 / MXFP8 experts.
- **Norm-constrained weights (optional)**: `--titan-row-norm-projection` rescales the embedding, attention, router and expert weight rows back to their released norms after every optimizer step.

## 2. Supported Variants

| Model | Active / Total | Layers | HF ID |
|---|---|---|---|
| Inkling 2 Mini | 5.4 B / 26 B | 24 (layers 6–17 looped × 3) | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) |

Architecture, from the checkpoint's `text_config`: hidden 2304, 24 attention heads with 8 KV heads, `head_dim=128`, 128 experts at `intermediate_size=1152`, vocab 201024 (200058 real tokens), 20 sliding-window and 4 full-attention layers, 6 layers per attention-residual block. The torchtitan model is `--titan-model-name inkling2 --titan-model-flavor looped`; it sizes itself from `config.json`, which may name either `inkling2` or `inkling2_loop`.

Training covers text prompts. The vision tower and the MTP head are not trained; the MTP head is carried into `--save-hf` exports unchanged, so an exported checkpoint still serves with speculative decoding.

## 3. Environment Setup

Use a Miles image whose SGLang supports Inkling 2 Mini.

`scripts/run_inkling2_mini.py` downloads the model and DAPO-Math-17k, then dequantizes the release checkpoint to BF16 once, into `{model_dir}/Inkling-2-Mini-bf16`:

```bash
python tools/convert_inkling2_to_bf16.py \
  --model-dir /root/models/Inkling-2-Mini --save-dir /root/models/Inkling-2-Mini-bf16
```

The conversion rewrites `hf_quant_config.json` so SGLang serves the BF16 weights, and copies the tokenizer, chat template and `mtp.safetensors` unchanged.

## 4. Launch

The validated recipe is full-parameter GRPO on DAPO-Math-17k on one node of 8× B200: 32 prompts × 8 samples per step, 4,096-token responses, Adam at lr 1e-6.

```bash
cd /root/miles
python scripts/run_inkling2_mini.py
```

`--num-rollout`, `--model-dir`, `--data-dir` and `--num-gpus-per-node` override the defaults; anything else goes in `--extra-args`, for example a final HF export:

```bash
python scripts/run_inkling2_mini.py --extra-args "--save-hf /root/ckpt/inkling2-mini/hf_{rollout_id} --save-interval 100"
```

## 5. Recipe Configuration

### 5.1 Parallelism

| Knob | Value |
|---|---|
| Trainer | FSDP over all GPUs, expert parallel = GPUs per node |
| Tensor / pipeline / context parallel | 1 / 1 / 1 |
| Rollout engine | one TP1 engine per GPU, colocated |
| `--seq-length` | 6144 (prompt + 4,096-token response) |

### 5.2 Optimizer

Adam at lr 1e-6, constant schedule, weight decay 0.1. `--optimizer muon` builds torchtitan's DistMuon with the model's own layout: dense projections orthogonalized whole, routed experts one matrix per expert, the rest on AdamW; Muon needs `--no-offload-train`, because offloading moves the parameter storage DistMuon holds on to.

### 5.3 Checkpoints

`--save-hf <dir>` writes a servable HF checkpoint: the source checkpoint with the trained tensors replaced, every frozen tensor and side file (`mtp.safetensors`, tokenizer) copied. It skips the torchtitan DCP checkpoint, so a run saved this way cannot resume its optimizer state.

### 5.4 What a healthy run looks like

On the recipe above, over 100 steps:

- `rollout/raw_reward` about 0.52 over the first 10 steps, about 0.85 over the last 10. Most of the gain is responses finishing inside 4,096 tokens: `rollout/truncated_ratio` falls from about 0.47 to 0.09.
- `train/train_rollout_kl` 0.7–1.2e-3
- `train/train_rollout_logprob_abs_diff` 0.012–0.018, flat

A log-prob gap that grows across weight syncs means the engines are not serving what the trainer trained.

### 5.5 Notable quirks

- The checkpoint's `config.json` sets `model_max_length`; a `--seq-length` above it is rejected at startup.
- Variable-pass loops (`loop_config.is_varloop: true`) are rejected; the release checkpoint uses a fixed three passes.
- `--rm-type deepscaler` accepts Inkling's `<|content_text|>` answer delimiter as well as `</think>`.

## 6. Pairs Well With

- [Inkling 2 Small](/models/thinkingmachines/inkling-2-small)
- [torchtitan backend](/user-guide/training-backend)
