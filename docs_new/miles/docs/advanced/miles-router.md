---
title: Miles Router (R3)
description: The proxy in front of SGLang that captures expert routing for MoE replay.
---

# Miles Router & R3

Miles Router is a small FastAPI proxy that sits between Miles and one or more SGLang
worker servers. It exists for one reason: **MoE RL is unstable unless training and
inference make the same expert routing decisions** — and the only way to guarantee that
is to capture routing on the inference side and replay it on the training side.

## What Miles Router does

* Registers SGLang workers in a pool.
* Routes requests with simple least-inflight load balancing.
* Health-checks workers and quarantines unhealthy ones.
* **Preserves response metadata** end-to-end (this is the key part).
* Hosts middleware plugins for cache, request/response transforms, etc.

It's the rollout-side of Miles ("SGLang + router") that pushes samples into the data
buffer.

## How it gets launched

In a distributed Miles job:

* If `--use-miles-router` is set → Miles starts Miles Router.
* Otherwise → Miles starts the SGLang Model Gateway.

If you set `--sglang-router-ip`, Miles connects to an external router instead.

## Why we need it

Production inference doesn't care about token-level logprobs or expert routing decisions.
RL rollout does. Specifically:

| RL needs | Why |
|---|---|
| Token-level logprobs | For PPO/GRPO loss computation |
| Loss masks | To exclude tool/observation tokens from gradient |
| **Expert routing** | For R3 — bit-exact MoE replay |

The SGLang Model Gateway is a Rust router optimised for serving — it reconstructs
responses with a fixed schema and **drops anything it doesn't recognise**, including
the routing metadata Miles needs. Miles Router does straight passthrough.

## R3 — Rollout Routing Replay

> Rollout Routing Replay (R3) records the expert routing decisions made during inference
> and replays them during training. The result is bit-wise identical expert allocation
> between rollout and training.

### Why MoE RL was previously broken

For each token, an MoE router picks `top-k` experts. The choice depends on inputs through
a soft router → top-k op. In production, the router is a learned `nn.Linear` with
non-deterministic kernels and FP8 quantisation; tiny numerical differences **flip
routes** at the per-layer, per-token level.

Without R3:

* Rollout selects experts `{2, 7}` for token 314.
* Training (with the same weights but slightly different precision/kernels) selects
  experts `{2, 8}` for token 314.
* Gradient is computed against the *wrong* expert. Multiplied by 100 layers × 32 K
  tokens × thousands of training steps → divergence.

With R3 the inference router's choice is what training also uses. Numerical noise no
longer flips routes.

### How R3 wires up

**SGLang side:**

* `--enable-return-routed-experts` — enables capture.
* `RoutedExpertsCapturer` records `topk_ids` at every MoE layer.
* `return_routed_experts=true` request param triggers the capture.
* Response includes `routed_experts` in `meta_info` — shape
  `[seq_len-1, num_layers, top_k]`.

**Miles side:**

* Set both `--use-miles-router --use-rollout-routing-replay`.
* Rollout sends `return_routed_experts=true` and stores results in
  `sample.rollout_routed_experts`.
* Trainer calls `fill_routing_replay()` to load routing into `RoutingReplay` objects.
* During forward pass, recorded routes are replayed instead of recomputed.

### Performance

R3 adds:

* **Rollout:** ~3% overhead (extra metadata in response).
* **Training:** negligible — replaying a route is a cheap index op.
* **Memory:** ~2 bytes/(token × layer × top-k). For 32 K tokens × 60 layers × 8 = 30 MB
  per sample.

### Verifying R3 is active

When R3 is correctly wired, you should see in the trainer logs:

```text
router/replay_hit_rate = 1.000
moe/expert_balance_std = 0.04
```

A `replay_hit_rate < 1.0` means Miles Router didn't preserve routing in some responses
— almost always a misconfigured passthrough.

## Radix-tree cache (transparent token management)

Use this when your rollout pipeline is **text-in / text-out** and you can't reliably
persist token IDs. (If you already control token-in / token-out — like Search-R1 or the
multi-turn VLM example — you don't need it.)

The problem it solves: re-tokenising response text at training time often produces
different tokens than rollout produced, breaking PPO/GRPO per-token alignment.

The cache:

* Intercepts text-based requests.
* Tokenises them once.
* Stores `(text, token_ids, logprobs, loss_masks)` in a radix tree keyed by text prefix.
* Allows `/retrieve_from_text` to return the exact token sequence with aligned metadata.
* Periodically cleans up stale nodes.

Useful for GRPO with multiple trajectories sharing a prompt prefix.

## Miles Router vs. SGLang Model Gateway

| | Miles Router | SGLang Model Gateway |
|---|---|---|
| Implementation | Python / FastAPI | Rust |
| Goal | Preserve everything | Maximise serving throughput |
| Schema | Passthrough | Fixed (drops unknown) |
| Routing metadata | ✅ Preserved | ❌ Dropped |
| Cache-aware routing | Basic (least-inflight) | Advanced |
| Circuit breakers / retries | Basic | Advanced |
| PD disaggregation | – | ✅ |

Use Miles Router when you need R3 or radix caching. Use SGLang Model Gateway when you
want raw serving performance and don't need RL metadata.

→ [SGLang Model Gateway docs](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)

## Middleware

Plug request/response middleware via `--miles-router-middleware-paths`:

```bash
--miles-router-middleware-paths \
   my_pkg.middleware.add_request_id,my_pkg.middleware.cache_responses
```

Each middleware is a callable that receives the request/response and can mutate, log, or
short-circuit. Use this for:

* Audit logging
* Custom caching policies
* Request fingerprinting
* Latency injection during chaos tests

## When you can skip R3

* You're not training an MoE.
* You're using `--advantage-estimator reinforce++` *and* you already use TIS to mask the
  off-policy term.
* You've measured `expert_balance_std < 0.05` without R3 (rare).

Otherwise, on MoE: keep R3 on. It's free.

## References

* R3 paper — [arXiv 2510.11370](https://arxiv.org/pdf/2510.11370)
* Routing replay — [arXiv 2507.18071](https://arxiv.org/abs/2507.18071)
