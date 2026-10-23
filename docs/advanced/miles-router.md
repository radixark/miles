---
title: Rollout Routing Replay (R3)
description: Capture expert routing during inference and replay it during training to stabilize RL.
---
Rollout Routing Replay (R3) records the expert routing decisions made during
inference and replays them during training, producing bit-identical expert
allocation between rollout and training.

## Why MoE RL is unstable without R3

For each token, an MoE router picks `top-k` experts. The choice depends on the
input through a soft router and a top-k operation. In production the router is a
learned `nn.Linear` with non-deterministic kernels and FP8 quantization, so tiny
numerical differences flip routes at the per-layer, per-token level.

An example without R3:

* Rollout selects experts `{2, 7}` for token 314.
* Training (with the same weights but slightly different precision and kernels)
  selects experts `{2, 8}` for token 314.
* The gradient is computed against the wrong expert. Multiplied by hundreds of
  layers, tens of thousands of tokens, and thousands of training steps, the
  policy diverges.

With R3, the trainer replays the rollout router's expert assignments verbatim,
so numerical noise no longer flips routes.

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

## Routing replay forcing for multi-turn sessions

In a multi-turn session, a `retract` weight sync, a cache flush or a radix-cache
miss makes SGLang re-prefill tokens it has already served. Under newer weights
the recomputed routes can differ from the first ones, so R3 can train those
tokens with routes that did not sample them.

`--use-rollout-routing-replay-forcing` makes each engine keep, per session, the
expert ids it first computed for every token and reuse them on every recompute;
gate weights and KV are still recomputed under the current weights. With forcing
on, the session server requests only new R3 rows even under `retract`.

```bash
--use-rollout-routing-replay --use-session-server --use-rollout-routing-replay-forcing \
--sglang-moe-runner-backend triton
```

- A session's turns must reach one engine: use the session-server default router
  policy (`manual`) or `consistent_hashing`, without `--use-miles-router`.
- Only the trained model's engines run forcing; the first weight sync fails if
  one of them does not report `routed_experts_forcing` in `/server_info`. SGLang
  refuses to start forcing with options it cannot support, such as a MoE runner
  other than `triton`, `flashinfer_cutlass` or `flashinfer_trtllm_routed`, data
  or pipeline parallelism, or speculative decoding.
- The engine fails closed: a turn whose routes it no longer holds is refused
  before it runs, the session returns no sample, and the rollout drops the group
  like any group with an aborted sample. `rollout/r3/forcing_refused_turns`
  counts refusals; `rollout/r3/forced_turn_frac` and
  `rollout/r3/retracted_turn_frac` show how often turns were forced and retracted.
- Held routes live in host memory on every TP rank, capped by
  `--sglang-routed-experts-forcing-max-tokens` with LRU eviction (the default is
  4M tokens, about 1.5 GiB per rank for Qwen3-30B-A3B). Size it for an engine's
  live sessions: an evicted session is refused on its next turn.
- Each session gets its own radix-cache namespace, so sessions no longer share
  prompt KV.

## Memory cost

`(num_tokens - 1) × num_layers × top_k × 4 bytes` (int32 per element, see
`miles/utils/types.py`). For a 32K-token sequence, 60 layers, and
`top_k = 8`, that is roughly 60 MB per sample of routing metadata.

## When R3 is not required

* The model is dense.
* The `--advantage-estimator` is `reinforce_plus_plus` and `--use-tis` already
  masks the off-policy term.

For MoE training with `--advantage-estimator grpo`, current recipes turn R3
on (for example `scripts/run_glm47_flash.py`).

## References

* R3 paper: [arXiv 2510.11370](https://arxiv.org/pdf/2510.11370).
* Routing replay: [arXiv 2507.18071](https://arxiv.org/abs/2507.18071).
