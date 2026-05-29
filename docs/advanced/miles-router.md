---
title: Rollout Routing Replay (R3)
description: Capture expert routing during inference and replay it during training so MoE RL is stable.
---

# Rollout Routing Replay (R3)

Rollout Routing Replay (R3) records the expert routing decisions made during
inference and replays them during training, producing bit-identical expert
allocation between rollout and training.

## Why MoE RL was previously unstable

For each token, an MoE router picks `top-k` experts. The choice depends on the
input through a soft router and a top-k op. In production the router is a
learned `nn.Linear` with non-deterministic kernels and FP8 quantization, so tiny
numerical differences flip routes at the per-layer, per-token level.

Without R3:

* Rollout selects experts `{2, 7}` for token 314.
* Training (with the same weights but slightly different precision and kernels)
  selects experts `{2, 8}` for token 314.
* The gradient is computed against the wrong expert. Multiplied by hundreds of
  layers, tens of thousands of tokens, and thousands of training steps, the
  policy diverges.

With R3 the inference router's choice is what training also uses. Numerical
noise no longer flips routes.

## How R3 wires up

**SGLang side.** Miles enables `enable_return_routed_experts` automatically when
`--use-rollout-routing-replay` is on. SGLang then includes `routed_experts`
in `meta_info` of each response, with shape
`(seq_len - 1, num_layers, top_k)` and dtype `int32`.

**Miles side.** Enable R3 with:

```bash
--use-rollout-routing-replay
```

Rollout sends `return_routed_experts=true` in each request and stores the
results in `sample.rollout_routed_experts` (`miles/utils/types.py`). The
trainer pushes the arrays through `RoutingReplayManager`
(`miles/utils/replay_base.py`), and `replay_utils.py` plugs them into the
forward pass so recorded routes are used instead of recomputed ones.

## Memory cost

`(num_tokens - 1) × num_layers × top_k × 4 bytes` (int32 per element, see
`miles/utils/types.py:29`). For a 32K-token sequence, 60 layers, and
`top_k = 8`, that is roughly 60 MB per sample of routing metadata.

## When R3 is not required

* The model is dense.
* The `--advantage-estimator` is `reinforce_plus_plus` and `--use-tis` already
  masks the off-policy term.

For MoE training with `--advantage-estimator grpo`, current recipes turn R3
on (for example `scripts/run-glm4.7-flash.sh`).

## References

* R3 paper: [arXiv 2510.11370](https://arxiv.org/pdf/2510.11370).
* Routing replay: [arXiv 2507.18071](https://arxiv.org/abs/2507.18071).
