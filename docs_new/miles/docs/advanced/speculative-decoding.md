---
title: Speculative Decoding
description: Draft + target speculative rollout, with online SFT for MTP-style drafts.
---

# Speculative Decoding

Speculative decoding accelerates rollout by letting a lightweight draft model
generate ahead a few tokens and then verifying them with a single batched
forward of the target model. When the draft is correct the target produces N
tokens for the cost of one forward pass.

## Enabling speculative decoding

For models with built-in MTP (Multi-Token Prediction) layers (GLM-4.7,
DeepSeek-V3, DeepSeek-R1):

```bash
SGLANG_ARGS+=(
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
)
```

These are passthrough flags forwarded to SGLang. Miles auto-enables
`enable_draft_weights_cpu_backup` so SGLang can run training without
MTP weights resident on GPU
(`miles/backends/sglang_utils/sglang_engine.py`).

For an externally trained draft model (for example, trained with
[SpecForge](https://docs.sglang.ai/SpecForge/)):

```bash
SGLANG_ARGS+=(
   --sglang-speculative-draft-model-path /data/draft_model/
)
```

Full reference: [SGLang speculative decoding docs](https://docs.sglang.ai/advanced_features/speculative_decoding.html).

## Drift over a long RL run

As RL training progresses, the target model's distribution shifts away from the
draft. Fewer draft tokens pass verification, and over many steps speculative
decoding can become a net negative because the wasted draft compute outweighs
the verified speedup.

Miles supports training the draft alongside the target through online MTP-SFT.

## Online SFT for MTP-style draft models

```bash
PERF_ARGS+=(
   --mtp-num-layers 1
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

| Flag | Notes |
|---|---|
| `--mtp-num-layers` | Number of MTP layers in the checkpoint (1 matches GLM/DeepSeek release defaults). |
| `--enable-mtp-training` | Backprop through MTP loss alongside the policy loss. |
| `--mtp-loss-scaling-factor` | Weight of the MTP loss in the combined gradient (default `0.2`). |

<Note>

**Checkpoint must contain MTP weights.** Pass `--mtp-num-layers 1` when running `convert_hf_to_torch_dist.py`.
Without it the resulting `torch_dist` checkpoint will not contain the MTP
layer to train.

</Note>

## External draft model SFT

Training an external (non-MTP) draft model online is not yet supported in
Miles. The current path is to retrain the external draft offline every N
rollouts and reload it.

## Pairs with

* [Unified FP8](fp8-low-precision.md). Draft and target both quantized the
  same way.
* [INT4 QAT](int4-qat.md). A quantized draft is cheaper to verify.
* [R3](miles-router.md). R3 captures routing for the verified tokens emitted
  by the target.

## When to skip

* Rollout-bound on dense models below ~13 B. The verification overhead can
  outweigh the benefit.
* Already at high draft acceptance and the bottleneck is verification compute,
  not generation.

## Reading

* SpecForge: [SGLang docs](https://docs.sglang.ai/SpecForge/).
