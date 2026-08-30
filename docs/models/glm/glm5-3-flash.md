---
title: GLM-5.3-Flash
description: Text and vision-language RL recipes for GLM-5.3-Flash, a KDA + DSA hybrid MoE with mHC hyper-connections and NoPE MLA.
---

Implementation: text support is in
[`radixark/miles#2786`](https://github.com/radixark/miles/pull/2786), with vision-language
support stacked in [`radixark/miles#2792`](https://github.com/radixark/miles/pull/2792). It goes with the SGLang
[`sglang-miles-glm53next`](https://github.com/sgl-project/sglang/tree/sglang-miles-glm53next)
branch and [`radixark/Megatron-LM#89`](https://github.com/radixark/Megatron-LM/pull/89); the
image in section 3 pins all three.

## 1. Model Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`model_type: glm5_next`) is a
**45-layer KDA + DSA hybrid MoE**. It is a different architecture from the 744 B GLM5 and
GLM5.2 flagships, not a smaller cut of them.

- **45 layers, hybrid**: 34 KDA linear-attention layers + 11 DSA sparse-attention layers.
- **288-expert MoE**, sigmoid routing at top-8; the first 3 layers are dense.
- **mHC hyper-connections** at every block.
- **NoPE MLA** — multi-latent attention with the positional half of the QK head empty.
- **kpool-compressed lightning indexer** picks which keys the DSA layers attend.
- Hidden 4096, FFN 12288, 64 attention heads, vocab 154880, rotary base 800000.
- MTP is dropped for training.

## 2. Supported Variants

| Variant | `--model-name` | Layers |
|---|---|---|
| Full | `GLM-5.3-Flash` | 45 |
| 8-layer slice | `GLM-5.3-Flash-8layer` | 8 |
| 4-layer slice | `GLM-5.3-Flash-4layer` | 4 (launcher default) |

## 3. Environment Setup

Use `docker.io/radixark/miles:glm53next` — the rolling `radixark/miles:dev` image with the
three moving parts checked out at the versions this recipe was built against, multi-arch so
the same tag serves GB300 and x86 nodes.

| Component | Pinned at |
|---|---|
| miles | [`#2786`](https://github.com/radixark/miles/pull/2786) `1cd14c00` |
| SGLang | `sglang-miles-glm53next` `9a26e749` |
| Megatron-LM | [`#89`](https://github.com/radixark/Megatron-LM/pull/89) `e8f57451` |

```bash
hf download zai-org/GLM-5.3-Flash --local-dir /root/models/GLM-5.3-Flash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset chenhegu/geo3k_imgurl --local-dir /root/datasets/geo3k_imgurl
```

The reference checkpoint has to be converted first — `--ref-load` resolves to
`<--ckpt-dir>/<megatron_model_type>_torch_dist`:

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py glm5.3-flash)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
CONVERT_KEEP_PP1=1 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/root/Megatron-LM \
  torchrun --nproc-per-node 8 \
    tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/models/GLM-5.3-Flash \
    --save          /root/ckpt/glm5.3-flash_torch_dist
```

## 4. Launch

Bring up a ray cluster across the nodes, `export MILES_SCRIPT_EXTERNAL_RAY=1`, then on the
head node:

```bash
cd /root/miles
python scripts/run_glm5_3_flash.py train \
  --model-name GLM-5.3-Flash \
  --num-nodes 16 --num-gpus-per-node 4 \
  --num-rollout 20 --rollout-max-response-len 4096
```

Smoke slice on one node:

```bash
python scripts/run_glm5_3_flash.py train --num-nodes 1 --num-gpus-per-node 8
```

Vision-language smoke on Geo3K uses the same converted language checkpoint and
the visual weights in the original Hugging Face checkpoint:

```bash
python scripts/run_glm5_3_flash.py train \
  --task geo3k \
  --num-nodes 1 --num-gpus-per-node 8 \
  --rollout-batch-size 2 --n-samples-per-prompt 2 \
  --global-batch-size 4 --rollout-max-response-len 512
```

The Geo3K path uses the model's native dynamic image resize and patch order in
both Miles and SGLang. On the Megatron side, the original Hugging Face visual
tower is loaded identically on the embedding pipeline stage, kept frozen, and
its merged patch embeddings replace the checkpoint's image-token positions.
Only the already validated language parameters are converted, optimized,
checkpointed, and synchronized. The reduced-layer checkpoints therefore must
retain the full `model.visual.*` tensor set as well as `vision_config`.
Because that frozen tower is deliberately excluded from the optimizer and the
normal language-weight stream, the B300 VLM recipe offloads only SGLang's KV
cache and keeps its weights resident. SGLang loads the same Hugging Face visual
tower at startup; only the language weights change during training. Validate
the more expensive full-weight offload mode on the target topology before
production use.

For B300 (SM103), set `NCCL_NVLS_ENABLE=0`, keep Megatron's attention backend on
`auto`, and use SGLang's `sdpa` multimodal attention backend. Do not select the
Hopper-only FA3 path. The validated environment applies the KDA portion of
upstream FLA commit `3c4c54ae`, which hoists `triton.next_power_of_2` out of the
KDA JIT kernel for Triton 3.7; remove that compatibility patch once the pinned
FLA release contains the upstream fix.

| Shape | TP | PP | EP | Rollout engine |
|---|---|---|---|---|
| 16 × 4 (full, text validated) | 8 | 4 | 16 | 8 GPUs, SGLang TP 8 / EP 8 |
| 8 × 4 (full) | 8 | 4 | 16 | 8 GPUs, SGLang TP 8 / EP 8 |
| 6 × 4 (full) | 8 | 3 | 8 | 8 GPUs, SGLang TP 8 / EP 8 |
| 2 × 4 or 1 × 8 (slices; 1 × 8 VLM smoke validated) | 2 | 2 | 2 | 4 GPUs, SGLang TP 4 / EP 4 |

The PP-4 shapes run 11 / 11 / 11 / 12 layers per stage, since 45 does not divide by 4.
GRPO on DAPO-Math-17k, Adam at `lr 1e-6`, `max_tokens_per_gpu 8192`, full uniform recompute.
Rollout is colocated, with the trainer offloaded to disk; both DSA paths run on tilelang and
the KV cache is BF16. Routing replay is wired end to end, and indexer-topk replay
(`--use-rollout-indexer-replay`) is implemented but off by default.

## 5. What a Healthy Run Looks Like

From the validation run in [#2786](https://github.com/radixark/miles/pull/2786) — 16 nodes ×
4 GB300, DAPO on DAPO-Math-17k:

| Metric | Observed |
|---|---|
| `train/train_rollout_logprob_abs_diff` | 0.0068 – 0.0106 across the first 11 rollouts |
| `rollout/raw_reward` | 0.5 → 0.94 within 10 rollouts |
| `train/ppo_kl` | ~2.6e-4 |
| `train/grad_norm` | 0.31 – 0.49 |

`train/train_rollout_logprob_abs_diff` is the one to read first on a fresh bring-up: it
covers the KDA, DSA and hyper-connection paths at once.

The VLM validation in [#2792](https://github.com/radixark/miles/pull/2792) is a two-step
Geo3K smoke with the four-layer language slice and complete visual tower on one 8 × B300
node ([W&B run](https://wandb.ai/nan-playground/miles-glm53-vlm/runs/049apd1e)). Step 0 had
rollout/train mean log-probs of -9.5348/-9.5424, PPO KL 0.001155, and mean absolute
log-prob difference 0.08584; step 1 had -9.4927/-9.5067, PPO KL 0.001375, and mean
absolute difference 0.10136. The initial language-weight synchronization passed the exact
tensor checker, and both post-step synchronizations completed successfully. The tiny
language slice produced zero rewards, advantages, and gradient norm, as expected. This
validates image preprocessing, frozen-tower parity, rollout, training, and post-step
language-weight synchronization; it is not evidence for full 45-layer VLM convergence.
