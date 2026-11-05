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

## 2. Supported Variants

| `--model-name` | Layers | Purpose | GPUs |
|---|---|---|---|
| `Inkling2` | 48 | the release | 8 (one 8-GPU node, or two 4-GPU nodes), validated |

The name sets the checkpoint paths under `--model-dir`: `{model_dir}/Inkling2` for the HF weights and `{model_dir}/Inkling2_torch_dist` for the converted Megatron checkpoint. `--train-mode lora|full` picks the recipe; the launcher accepts only `lora` at full size, since full-parameter training does not fit 8 GPUs.

Architecture, from `scripts/models/inkling2.py`: hidden 4608, 48 attention heads with 8 query groups, `kv_channels=128`, 384 experts at `moe_ffn_hidden_size=2304`, vocab 201024, no position embedding. The model provider builds the rest from the checkpoint's `text_config`.

## 3. Environment Setup

Use the `radixark/miles:dev` image with the SGLang Inkling 2 Small support from `<sglang PR>`. A two-node conversion also needs `<Megatron-LM PR>` (see 3.2).

Download the model and the dataset:

```bash
hf download thinkingmachines/Inkling-2-Small --local-dir /root/models/Inkling2
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
```

### 3.1 HF → Megatron `torch_dist`, one 8-GPU node

`convert` loads the HF checkpoint through the Inkling 2 Small bridge, which dequantizes the NVFP4 / MXFP8 experts to BF16. It then writes `{model_dir}/Inkling2_torch_dist`, and skips the step if a finished conversion is already there.

```bash
cd /root/miles
python scripts/run_inkling2.py convert --model-name Inkling2 --train-mode lora --num-gpus-per-node 8
```

On one B300 node this takes about 28 minutes and writes 1.1 TB, peaking at about 170 GB per GPU while loading.

### 3.2 Two 4-GPU nodes

Join both nodes in one Ray cluster, `export MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head ip>`, and run on the head:

```bash
python scripts/run_inkling2.py convert --model-name Inkling2 --train-mode lora --num-nodes 2 --num-gpus-per-node 4
```

A synchronous `torch_dist` save stages each rank's whole shard in host memory by default: 4 × 138 GiB on a 626 GiB GB300 node, which is OOM-killed. `<Megatron-LM PR>` writes each tensor straight from the GPU instead, which keeps host staging under 1 GiB.

Each node needs the whole `Inkling2_torch_dist`, so put it on a filesystem both nodes read or copy it to each node.

## 4. Launch

The validated recipe is GRPO on DAPO-Math-17k: 32 prompts × 8 samples per step, 4,096-token responses, rank-32 LoRA, lr 5e-5. Adapter checkpoints are written every 20 steps.

One 8-GPU node (B300):

```bash
python scripts/run_inkling2.py train \
  --model-name Inkling2 --train-mode lora --num-nodes 1 --num-gpus-per-node 8 \
  --num-rollout 200 --rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256 \
  --rollout-max-response-len 4096 --sglang-context-length 8192 --lr 5e-5 \
  --extra-args "--offload-train-target cpu --check-lora-weight-equal --save /root/shared_data/inkling2-lora --save-interval 20"
```

Two 4-GPU nodes (GB300), on the head of the Ray cluster from 3.2:

```bash
python scripts/run_inkling2.py train \
  --model-name Inkling2 --train-mode lora --num-nodes 2 --num-gpus-per-node 4 \
  --num-rollout 200 --rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256 \
  --rollout-max-response-len 4096 --sglang-context-length 8192 --lr 5e-5 \
  --extra-args "--check-lora-weight-equal --save /scratch/inkling2-lora --save-interval 20"
```

The launcher offloads the trainer to local disk under `$SCRATCH` by default. That is the right choice on GB300, where host RAM holds the engines' base backup; on the B300, `--offload-train-target cpu` keeps the trainer in its 3 TB of host RAM instead. On node-local `--save` paths each node holds its own ranks' shards.

## 5. Recipe Configuration

### 5.1 Parallelism

| Knob | Value |
|---|---|
| Tensor / pipeline / context parallel | 1 / 1 / 1 |
| Expert parallel (trainer) | every GPU of the job (8) |
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

The engines serve the released NVFP4 / MXFP8 experts with W4A4 / W8A8 activation quantization; the trainer computes in BF16. That sets a train–rollout log-probability gap of about 0.019 at full size, flat over training. `--use-tis` (on in the recipe) corrects for it, and `--use-rollout-routing-replay` replays the rollout's expert choices in the trainer.

Full-parameter training fake-quantizes the trained expert weights to their layer's format with a straight-through gradient. Without that, each sync re-quantizes the experts and per-step updates below the NVFP4 / MXFP8 step are lost.

### 5.4 What a healthy run looks like

From the two-node GB300 run:

- `train/train_rollout_logprob_abs_diff` 0.018–0.021, flat (0.0202 at step 0)
- `train/train_rollout_kl` 1.1–1.4e-3, flat
- `rollout/raw_reward_unfiltered` 0.72 at step 0, about 0.85 by step 84
- about 10 minutes per step

A gap that grows across syncs while entropy is flat means the engine is not serving what the trainer trained: check `--check-lora-weight-equal` and that `--lora-base-cpu-backup` is on.

### 5.5 Notable quirks

- Multi-node runs set `NCCL_CUMEM_ENABLE=1`: cross-node NVLink (GB300 MNNVL) needs NCCL's cuMem buffers, and Miles otherwise disables them for the trainer.
- Some clusters mount `HF_HOME` read-only; point it at a writable directory before launching.
- The model lives in `miles_plugins/models/inkling2/`, the HF ↔ Megatron names in `miles/backends/megatron_utils/megatron_to_hf/inkling2.py` and `miles_plugins/mbridge/inkling2.py`, and the expert encoder for weight sync in `miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_inkling2.py`.

## 6. Pairs Well With

- [LoRA](/advanced/lora)
- [Inkling](/models/thinkingmachines/inkling)
- [Kimi-K3](/models/kimi/kimi-k3)
