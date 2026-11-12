---
title: Inkling 2 Small
description: LoRA RL recipe for Inkling 2 Small, a Block Attention Residuals MoE with NVFP4 / MXFP8 experts, trained colocated with SGLang.
---

## 1. Model Introduction

[Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) is the second generation of Thinking Machines Lab's Inkling models. It keeps Inkling's hybrid local and global relative attention and replaces the residual stream with **Block Attention Residuals**. The checkpoint ships its routed experts in **NVFP4 and MXFP8**, chosen per layer.

Miles implements it as a native Megatron model (`miles_plugins/models/inkling2/`). Every op keeps the serving kernel's rounding points, so the trainer computes the same function the rollout engine serves.

**Key highlights:**

- **Block Attention Residuals**: each layer mixes the completed residual blocks and the current partial block with a learned softmax over depth, instead of adding into one residual stream.
- **Relative attention**: sliding-window and full-attention layers, a learned relative-position bias, a gated output and a query-conditioned sink. Full-attention layers add dynamic K/V short convolutions.
- **Sigmoid-routed MoE**: 384 routed experts at top-8 with soft-capped SwiGLU; no shared experts.
- **The engine serves the release precision**: rollout uses the checkpoint's NVFP4 / MXFP8 experts, and weight sync re-encodes trained experts in the same per-layer format.
- **LoRA in the engine's adapter schema**: attention, shared-outer routed experts and the unembed, synced to the engines as adapters only.
- **Image prompts and agentic rollout**: multimodal RL through the frozen vision tower, and token-in/token-out multi-turn sessions.

## 2. Supported Variants

| `--model-name` | Layers | HF ID |
|---|---|---|
| `Inkling2` | 48 | [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) |

The name sets the checkpoint paths under `--model-dir`: `{model_dir}/Inkling2` for the HF weights and `{model_dir}/Inkling2_torch_dist` for the converted Megatron checkpoint. `--train-mode lora|full` picks the recipe.

Architecture, from `scripts/models/inkling2.py`: hidden 4608, 48 attention heads with 8 query groups, `kv_channels=128`, 384 experts at `moe_ffn_hidden_size=2304`, vocab 201024, no position embedding. The model provider builds the rest from the checkpoint's `text_config`.

## 3. Environment Setup

Use the `radixark/miles:inkling-2-small` image (amd64 and arm64). It carries the SGLang, Megatron-LM and Miles changes this page needs.

Download the model and the dataset:

```bash
cd /root/miles
hf download thinkingmachines/Inkling-2-Small --local-dir /root/models/Inkling2
python scripts/run_inkling2.py prepare        # DAPO-Math-17k under --data-dir
```

Convert HF → Megatron `torch_dist`. `convert` loads the HF checkpoint through the Inkling 2 Small bridge, which dequantizes the NVFP4 / MXFP8 experts to BF16, writes `{model_dir}/Inkling2_torch_dist`, and skips the step once a finished conversion is there:

```bash
python scripts/run_inkling2.py convert --model-name Inkling2 --train-mode lora \
  --num-nodes <nodes> --num-gpus-per-node <gpus>
```

On several nodes, join them in one Ray cluster, `export MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head ip>`, and run `convert` and `train` on the head. Every node needs the whole `Inkling2_torch_dist`.

## 4. Launch

The validated recipe is GRPO on DAPO-Math-17k: 32 prompts × 8 samples per step, 4,096-token responses, rank-32 LoRA, lr 5e-5, adapter checkpoints every 20 steps.

```bash
python scripts/run_inkling2.py train \
  --model-name Inkling2 --train-mode lora --num-nodes <nodes> --num-gpus-per-node <gpus> \
  --num-rollout 200 --rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256 \
  --rollout-max-response-len 4096 --sglang-context-length 8192 --lr 5e-5 \
  --extra-args "--check-lora-weight-equal --save <ckpt dir> --save-interval 20"
```

In the image, `inkling2-day0` runs the whole recipe: `prepare`, `convert`, then the `train` command above. Set `MODEL_DIR`, `DATA_DIR`, `SAVE_DIR`, `NUM_NODES` and `NUM_GPUS_PER_NODE` in the environment; extra launcher arguments go after the command.

## 5. Recipe Configuration

### 5.1 Parallelism

| Knob | Value |
|---|---|
| Tensor / pipeline / context parallel | 1 / 1 / 1 |
| Expert parallel (trainer) | every GPU of the job |
| Rollout engine | one per node, TP = EP = GPUs per node |
| `max_tokens_per_gpu` | 8192 |

Attention runs at TP1 because its short convolutions and attention-residual state are not sequence-sharded; experts shard over all GPUs instead.

### 5.2 LoRA

Rank 32 / alpha 32 by default, `--target-modules all-linear`, which resolves to the engine's adapter schema:

```
attn.wq_du, attn.wr_du, attn.gate_attn, attn.wk_dv, attn.wv_dv, attn.wo_ud
mlp.experts (shared-outer: one A for gate_up and one B for down, shared by all experts)
model.llm.unembed (every padded-vocab row)
```

The attention sink projection gets no adapter. The launcher always passes `--lora-base-cpu-backup`, so the engines keep the checkpoint's quantized base across pauses and each sync carries only adapters. Resyncing a base re-encoded from the trainer's BF16 copy would recompute the NVFP4 global scales.

### 5.3 Rollout and precision

The engines serve the released NVFP4 / MXFP8 experts with W4A4 / W8A8 activation quantization; the trainer computes in BF16. That sets a train–rollout log-probability gap of about 0.019, flat over training. `--use-tis` (on in the recipe) corrects for it, and `--use-rollout-routing-replay` replays the rollout's expert choices in the trainer.

Full-parameter training fake-quantizes the trained expert weights to their layer's format with a straight-through gradient. Without that, each sync re-quantizes the experts and per-step updates below the NVFP4 / MXFP8 step are lost.

### 5.4 Image prompts

`--task geo3k` trains on Geometry3K, whose prompts carry images:

```bash
python scripts/run_inkling2.py prepare --task geo3k   # downloads and converts once
python scripts/run_inkling2.py train --model-name Inkling2 --train-mode lora --task geo3k \
  --num-nodes <nodes> --num-gpus-per-node <gpus> --extra-args "--check-lora-weight-equal"
```

- **Rendering:** prompts stay structured messages, and SGLang's Inkling 2 Small renderer and image processor build the token ids and vision patches. The trainer therefore sees the same input the engine served.
- **Vision tower:** the trainer embeds images with SGLang's vision encoder on the checkpoint's `model.visual` weights. The tower is frozen and kept outside the module tree, so DDP, the optimizer, checkpoints and weight sync never see it.
- **Routing replay:** rows follow the prompt the engine served, in which each image placeholder is the image's whole token run.

### 5.5 Agentic rollout (TITO)

Multi-turn agentic rollout uses the Inkling family's TITO tokenizer, `--tito-model inkling` (see [Agentic Rollout](/user-guide/agentic-rollout)). Inkling 2 Small names no function on tool-call and tool-result headers, matching SGLang's renderer. Sessions over the OpenAI and Anthropic APIs, tool calls included, reproduce the engine's tokens exactly.

### 5.6 What a healthy run looks like

- `train/train_rollout_logprob_abs_diff` 0.018–0.021, flat
- `train/train_rollout_kl` 1.1–1.4e-3, flat
- `rollout/raw_reward_unfiltered` about 0.72 at step 0, rising to about 0.85 by step 85

A gap that grows across syncs while entropy is flat means the engine is not serving what the trainer trained: check `--check-lora-weight-equal` and that `--lora-base-cpu-backup` is on.

### 5.7 Notable quirks

- Multi-node runs set `NCCL_CUMEM_ENABLE=1`: cross-node NVLink needs NCCL's cuMem buffers, and Miles otherwise disables them for the trainer.
- Some clusters mount `HF_HOME` read-only; point it at a writable directory before launching.
- The model lives in `miles_plugins/models/inkling2/`, the HF ↔ Megatron names in `miles/backends/megatron_utils/megatron_to_hf/inkling2.py` and `miles_plugins/mbridge/inkling2.py`, and the expert encoder for weight sync in `miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_inkling2.py`.

## 6. Pairs Well With

- [LoRA](/advanced/lora)
- [Inkling](/models/thinkingmachines/inkling)
- [Kimi-K3](/models/kimi/kimi-k3)
