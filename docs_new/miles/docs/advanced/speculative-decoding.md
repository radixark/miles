---
title: Speculative Decoding
description: Draft + target speculative rollout, with online SFT for the draft.
---

# Speculative Decoding

Speculative decoding is the cheapest way to make rollout faster. A lightweight **draft
model** decodes ahead a few tokens; the **target model** then verifies them in a single
batched forward. When the draft is right (~70–90% of the time on most workloads), you
get N tokens for the cost of one forward.

## Enabling speculative decoding

For models with built-in MTP (Multi-Token Prediction) layers — GLM-4.7, DeepSeek-V3,
DeepSeek-R1 — turn it on with:

```bash
SGLANG_ARGS+=(
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   --sglang-enable-draft-weights-cpu-backup
)
```

For an externally trained draft model (e.g. trained with
[SpecForge](https://docs.sglang.ai/SpecForge/)):

```bash
SGLANG_ARGS+=(
   --sglang-speculative-draft-model-path /data/draft_model/
)
```

Full parameter reference: [SGLang speculative decoding docs](https://docs.sglang.ai/advanced_features/speculative_decoding.html).

## The drift problem

As RL training progresses, the **target** model's distribution shifts away from the
**draft**'s. Fewer draft tokens pass verification, and after thousands of steps
speculative decoding can become a *net negative* — the wasted draft compute outweighs
the verified speedup.

The fix: train the draft alongside the target.

## Online SFT for MTP-style draft models

Miles supports online training of the MTP layers during RL — the draft model evolves with
the target.

```bash
PERF_ARGS+=(
   --mtp-num-layers 1
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

!!! note "Checkpoint must contain MTP weights"
    Pass `--mtp-num-layers 1` when you run `convert_hf_to_torch_dist.py`. Without it the
    `torch_dist` checkpoint won't have the MTP layer to train.

| Knob | What |
|---|---|
| `--mtp-num-layers 1` | One MTP layer (matches GLM/DeepSeek release defaults) |
| `--enable-mtp-training` | Backprop through MTP loss alongside the policy loss |
| `--mtp-loss-scaling-factor 0.2` | Don't let MTP loss dominate the policy gradient |

## What you should see

After enabling online MTP-SFT, watch:

```text
spec/draft_acceptance_rate     stable around 0.70–0.85
spec/avg_accepted_tokens       2.5 – 3.0 (out of 4 drafted)
spec/effective_rollout_speedup 1.7× – 2.3×
mtp/loss                       slowly decreasing
```

If acceptance rate is collapsing, either MTP loss scaling is too low or the draft
needs more training data.

## External draft model SFT

Training an external (non-MTP) draft model online is **work in progress**. Until then,
re-train the external draft offline every N rollouts and reload it.

## Combine with everything else

Speculative decoding stacks cleanly with:

* **[FP8 unified](fp8-low-precision.md)** — draft and target both quantised the same
  way.
* **[INT4 QAT](int4-qat.md)** — quantised draft is ~4× cheaper to verify.
* **[R3](miles-router.md)** — R3 captures routing for the *target* (verified) tokens.

## When to skip

* You're rollout-bound on dense models < 13 B — overhead outweighs benefit.
* You're already at maximum draft acceptance and the bottleneck is verification compute,
  not generation.

## Reading

* SpecForge — [SGLang docs](https://docs.sglang.ai/SpecForge/)
* Power-Up Spec Decoding in RL — [blog](https://www.notion.so/jiajunli-guapisolo/Power-Up-Speculative-Decoding-In-Reinforcement-Learning-2a92d24a293b802d9c73dbae429e581e)
