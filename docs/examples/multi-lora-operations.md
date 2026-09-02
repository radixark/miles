---
title: "Multi-LoRA operation backend with Tinker compatibility"
description: "Multi-adapter LoRA trained through explicit operations; stacked PR #2346 provides Tinker REST/SDK compatibility."
# Generated from examples/multi_lora_operations/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Serve many LoRA training runs on one shared base model through the
`MultiLoraOperationBackend`. Clients drive training with explicit
`forward_backward` / `optim_step` operations — no dataset, reward function,
or batch schedule on the server. This PR exposes training operations through
the controller's Ray API; stacked PR #2346 supplies the
[tinker](https://tinker-docs.thinkingmachines.ai/)-compatible REST adapter.

```
internal caller ──Ray operations──> MultiLoraOperationBackend (head node)
                                      ├─ registration + operation ledger
                                      ├─ adapter-slot execution
                                      └─ serving plane ─────────> SGLang router

Tinker client ──HTTP──> stacked protocol adapter (#2346) ────────┘
trainer ranks <──Ray── driver loop (train_multi_lora_operations.py)
```

## Launch

```bash
python train_multi_lora_operations.py \
  --tinker-backend \
  --multi-lora-n-adapters 4 \
  --lora-rank 32 --lora-alpha 64 \
  --target-modules all-linear \
  --hf-checkpoint Qwen/Qwen3-0.6B \
  ... # the usual megatron/sglang flags; see run_multi_lora_operations.py
```

Key flags:

| flag | meaning |
|------|---------|
| `--tinker-backend` | enable the Tinker protocol adapter for the Multi-LoRA operation backend (requires `--multi-lora-n-adapters > 0`) |
| `--multi-lora-n-adapters N` | fixed slot count; a registration binds a slot for life (queue when full) |
| `--lora-rank` / `--lora-alpha` | deployment-wide ceiling / fixed alpha — clients may lower `rank`, never set `alpha` |
| `--multi-lora-api-port` | control-plane API port for runtime adapter registration |
| `--tinker-max-coalesce-wait-s` | how long one train call coalesces additional ready client batches |
| `--tinker-max-empty-wait-s` | idle-queue yield back to the control phase (keep this small) |

### Activation recompute (memory saving)

`--recompute-granularity selective` is always supported (default
`--recompute-modules core_attn`; add `moe_act` to also recompute the MoE
activation with grouped GEMM). `--recompute-granularity full` — and `moe` in
`--recompute-modules` when expert modules are targeted — is supported as
well: the deployment's Megatron-Bridge (branch `bridge`) recognizes
multi-LoRA `.adapters.<slot>.` params in its PEFT recompute patch, so
checkpointed regions replay grad-enabled during adapter-only training.

## Operation contract

`Tinker` names the compatibility boundary, not the trainer implementation.
The current concrete is `MultiLoraOperationBackend`; its queue-backed
`MultiLoraOperationBatchFn` batches already-tokenized operations, and the
Megatron `MultiLoraParameterExecutor` applies them to adapter slots. A future
full-parameter composition can reuse the same operation contract and the
unwired `FullParameterExecutor` sibling; full-parameter launch, data-path,
checkpoint, and publish integration are not implemented by this stack today.

```
Tinker protocol frontend
          │
          ▼
generic training-operation contract
          │
          ├── MultiLoraParameterExecutor      (current wired target)
          └── FullParameterExecutor           (implemented seam; not wired)
```

`enqueue_operation(name, operation_id, ordinal, kind, payload)` — ordinals are
consecutive per registration starting at 1; arrival may be out of order
(gap-buffered, and a hole-filling ordinal is always admitted), execution is
strictly ordinal-ordered; retries with the same `operation_id`, same ordinal,
and identical payload return the original operation — anything else is a
typed conflict.

A stream stalled on a never-arriving ordinal (the 0.24.1 SDK consumes a
seq_id and can then fail BEFORE HTTP — see the SDK limitations below) expires
after `--tinker-operation-gap-timeout` (default 600 s, `<= 0` disables): the
blocked, never-claimed operations terminal-fail `FAILED(user)` naming the
missing ordinal, and the hole is sealed — the missing identity never executes
(a late arrival is a typed conflict), nothing overtakes it, and the client
resubmits as new operations. Stalls are observable before expiry:
`service_info()` reports `gap_stalls`, and a blocked operation's
`get_operation` view carries `waiting_on_ordinal` / `gap_stalled_for`.

| kind | payload | success result |
|------|---------|----------------|
| `forward_backward` | `{samples: [Datum...], loss: {loss_fn, loss_fn_config?}}` | `{logprobs: [[...]], metrics: {"loss:sum", "unmasked_tokens:sum", "loss_weight:sum" (CE only)}}` |
| `forward` | `{samples: [Datum...]}` | `{logprobs: [[...]]}` (zero gradient, structurally) |
| `optim_step` | `{adam_params: {learning_rate, beta1, beta2, eps, weight_decay, grad_clip_norm}}` | `{grad_norm, learning_rate}` |
| `save_weights_for_sampler` | `{}` | `{serving_version, serving_name}` — completes only after the weights are live |
| `save_state` | `{tag?, ttl_seconds?}` | `{path, step}` (named states are immutable) |
| `load_state` | `{path}` | `{step, path}` (re-publishes on the next push) |

`Datum = {tokens, response_length, loss_mask, loss_weights?, advantages?, rollout_log_probs?}`
— per-token channels align with the response span. Losses reduce as plain
token sums (`Σ(-logp·w)` for `cross_entropy`), so K chunked forward_backward
calls accumulate exactly like one; `loss_weights` own the scale and no server
normalization or scheduler ever touches a tinker slot. Result `metrics` use
the SDK combiner's `name:reduction` keys.

For an SFT-style per-token loss, divide `loss:sum` by `loss_weight:sum`
(cross-entropy only: Σ weight·mask, chunk-additive like the loss) — NOT by
`unmasked_tokens:sum`, which counts every loss_mask-active position and so
includes the weight-0 prompt tokens of a teacher-forced datum, silently
diluting the displayed loss. Guard the division: weights are arbitrary
finite floats, so the sum can be zero or negative.

Operation states: `QUEUED → CLAIMED → SUCCEEDED | FAILED(user|server) | CANCELLED`;
poll `get_operation`, then `ack_operation` to release the record. In v1 these
verbs are the controller actor's Ray API (registration/status are the only
HTTP routes); backpressure raises a retryable `OperationBackpressure` — the
future tinker HTTP frontend maps it to 429 + Retry-After, never to a 4xx the
SDK treats as fatal. Deregistering fences every open operation of that
registration as `FAILED(user)`.

## v1 compatibility matrix

Supported: text-only input; the synchronous training loop; 1-D shifted
targets; `loss_fn ∈ {cross_entropy, importance_sampling, ppo}` (per-op clip
config); per-call AdamParams; multi-chunk gradient accumulation with
independent `optim_step`; latest-only sampler weights behind the publish
barrier; named immutable `save_state` / `load_state` (create-from-checkpoint
included, shape-fenced); optional `num_step` auto-retirement.

Explicitly rejected (boundary error, never a silent fallback): multimodal
inputs; nested `(N, K)` top-K targets; other loss functions (CISPO, DRO, ...);
client-set `alpha`; non-finite/out-of-domain AdamParams; a loss's required
per-token channels missing; `response_length == len(tokens)` (targets are
shifted); async/off-policy sampling against pinned snapshots;
cross-world-size state restore; state restore into a slot whose per-rank
optimizer ownership differs from the save (cross-slot restore requires an
identical dense-and-expert ownership signature); idle slot GC.

## Known tinker SDK (0.24.1) client-side limitations

The official `tinker==0.24.1` TrainingClient takes its per-model seq counter
BEFORE it serializes and POSTs a request, so a submission can die client-side
with the ordinal already spent (verified against the live stack,
codex-0817-sft-fix §4-§6):

- **Pre-HTTP serialization failure** — e.g. `AdamParams(learning_rate=nan)`
  raises a local JSON `ValueError`; the request never reaches Miles and later
  operations of the same client queue behind the hole. The gap timeout above
  terminal-fails them typed, and the SAME TrainingClient can resubmit
  afterwards (its turn counter did advance). Validate that Adam params and
  custom scalars are finite before calling the SDK to avoid the stall.
- **`.future().cancel()` on an SDK future** can spend the request id without
  advancing the SDK's internal turn counter: later operations of that client
  wait forever CLIENT-side and Miles receives nothing it could terminalize —
  no server-side mitigation exists. Do not cancel underlying SDK futures;
  `.result(timeout=...)` is safe (non-destructive, the future stays
  retrievable). After an immediate cancel, discard the TrainingClient and
  create a new one (a fresh registration). When some submissions did reach
  the server, the gap timeout converts the surviving stall into typed
  failures instead of a hang.
- The server never skips a missing ordinal and never guesses what it would
  have been: the gap timeout only fails what is blocked and seals the hole,
  so strict per-registration ordering, idempotent retries, and anti-replay
  all hold.

## Files

- `run_multi_lora_operations.py` — disaggregated service launch (`prepare` / `serve`)
