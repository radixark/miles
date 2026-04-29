---
title: MilesRouter and R3
description: The proxy in front of SGLang that captures expert routing for MoE replay.
---

# MilesRouter and R3

MilesRouter is a FastAPI proxy that sits between Miles and one or more SGLang
worker servers. The motivation is that MoE RL is unstable when training and
inference disagree on expert routing decisions. The most reliable way to keep
them in sync is to capture routing on the inference side and replay it on the
training side. MilesRouter preserves the metadata needed for that path.

## What MilesRouter does

* Registers SGLang workers in a pool.
* Routes requests with simple least-inflight load balancing.
* Health-checks workers and quarantines unhealthy ones.
* Preserves response metadata end-to-end (the routing payload that the SGLang
  Model Gateway drops by design).
* Hosts middleware plugins for caching, request and response transforms, and
  custom routing.

## How it gets launched

In a distributed Miles job:

* If `--use-miles-router` is set, Miles starts MilesRouter.
* Otherwise, Miles starts the SGLang Model Gateway (the default `sglang_router`).

If you set `--sglang-router-ip` and `--sglang-router-port`, Miles connects to an
external router instead and skips starting its own. Engines register via
`/add_worker`.

## Why MilesRouter

Production inference does not need token-level logprobs or expert routing
decisions. RL rollout does:

| RL needs | Why |
|---|---|
| Token-level logprobs | For PPO/GRPO loss computation |
| Loss masks | To exclude tool/observation tokens from the gradient |
| Expert routing | For R3, bit-exact MoE replay |

The SGLang Model Gateway is a Rust router optimised for serving. It reconstructs
responses with a fixed schema and drops fields it does not recognise, including
the routing metadata Miles needs. MilesRouter does straight passthrough.

## R3: Rollout Routing Replay

> Rollout Routing Replay (R3) records the expert routing decisions made during
> inference and replays them during training, producing bit-identical expert
> allocation between rollout and training.

### Why MoE RL was previously unstable

For each token, an MoE router picks `top-k` experts. The choice depends on the
input through a soft router and a top-k op. In production the router is a
learned `nn.Linear` with non-deterministic kernels and FP8 quantisation, so tiny
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

### How R3 wires up

**SGLang side.** Miles enables `enable_return_routed_experts` automatically when
`--use-rollout-routing-replay` is on. SGLang then includes `routed_experts`
in `meta_info` of each response, with shape
`(seq_len - 1, num_layers, top_k)` and dtype `int32`.

**Miles side.** Set both:

```bash
--use-miles-router
--use-rollout-routing-replay
```

Rollout sends `return_routed_experts=true` in each request and stores results in
`sample.rollout_routed_experts` (`miles/utils/types.py`). The trainer pushes the
arrays through `RoutingReplayManager`
(`miles/utils/replay_base.py`), and `replay_utils.py` plugs them into the
forward pass so recorded routes are used instead of recomputed ones.

### Memory cost

`(num_tokens - 1) × num_layers × top_k × 4 bytes` (int32 per element, see
`miles/utils/types.py:29`). For a 32K-token sequence, 60 layers, and
`top_k = 8`, that is roughly 60 MB per sample of routing metadata.

## Radix-tree cache for `/generate`

Use this when the rollout pipeline is text-in / text-out and you cannot reliably
persist token IDs across the rollout-to-trainer boundary. Pipelines that already
control token-in / token-out (such as Search-R1 or the multi-turn VLM example)
do not need it.

The problem it solves: re-tokenising response text at training time often
produces different tokens than rollout produced, breaking PPO/GRPO per-token
alignment.

The cache:

* Intercepts text-based requests.
* Tokenises them once.
* Stores `string_key`, `token_ids`, `logp`, and `loss_mask` per node in a
  radix tree keyed by text prefix (`StringTreeNode` in
  `miles/router/middleware_hub/radix_tree.py`).
* Lets `/retrieve_from_text` return the exact token sequence with aligned
  metadata.
* Garbage-collects nodes older than `current_weight_version - gc_threshold_k`.

It is implemented as a middleware
(`miles.router.middleware_hub.radix_tree_middleware.RadixTreeMiddleware`).

## MilesRouter vs SGLang Model Gateway

| | MilesRouter | SGLang Model Gateway |
|---|---|---|
| Implementation | Python / FastAPI (`miles/router/router.py`) | Upstream `sglang_router` package |
| Goal | Preserve all response metadata | Production serving |
| Schema | Passthrough proxy | Fixed (drops unknown response fields) |
| Routing metadata | Preserved | Dropped |
| Routing policy | Minimum active requests (`_use_url`) | Configurable via `--sglang-router-policy` |
| PD disaggregation | Not supported (`assert` at `ray/rollout.py:930`) | Supported (`router_args.pd_disaggregation = True`) |

MilesRouter is required when you need R3, the radix-tree middleware, or custom
middleware. The SGLang Model Gateway covers the case where raw serving
throughput matters more than metadata preservation. Newer Miles recipes
(radix-tree, GLM-4.7-Flash, MoE training) run on MilesRouter, and more of the
recipe surface is moving over.

[SGLang Model Gateway docs](https://docs.sglang.io/advanced_features/sgl_model_gateway.html).

## Middleware

`--miles-router-middleware-paths` takes one or more middleware module paths
(space-separated, `nargs="+"`):

```bash
--miles-router-middleware-paths \
   my_pkg.middleware.add_request_id \
   my_pkg.middleware.cache_responses
```

Each middleware is a callable that receives the request and response and can
mutate, log, or short-circuit. Use cases include audit logging, custom caching
policies, request fingerprinting, and latency injection during chaos tests.

## When R3 is not required

* The model is dense.
* The `--advantage-estimator` is `reinforce_plus_plus` and `--use-tis` already
  masks the off-policy term.

For MoE training with `--advantage-estimator grpo`, current recipes turn R3
on (for example `scripts/run-glm4.7-flash.sh`).

## References

* R3 paper: [arXiv 2510.11370](https://arxiv.org/pdf/2510.11370).
* Routing replay: [arXiv 2507.18071](https://arxiv.org/abs/2507.18071).
