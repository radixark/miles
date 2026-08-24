---
title: "Multi-LoRA operation backend with Tinker compatibility"
description: "Multi-adapter LoRA trained through explicit operations with Tinker REST/SDK compatibility."
# Generated from examples/multi_lora_operations/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Serve many LoRA training runs on one shared base model through the
`MultiLoraOperationBackend`. The
[tinker](https://tinker-docs.thinkingmachines.ai/)-compatible frontend maps the
official SDK onto explicit `forward_backward` / `optim_step` operations and
shared-engine sampling — no dataset, reward function, or batch schedule on the
server.

```
official Tinker SDK ──HTTP──> Tinker protocol/frontend adapter
                                  │
internal caller ──Ray operations───────┘
                                  ▼
                    MultiLoraOperationBackend (head node)
                         ├─ registration + operation ledger
                         ├─ adapter-slot execution
                         └─ serving plane ─────────> SGLang router
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
| `--tinker-frontend` | serve the official tinker SDK REST protocol (`/api/v1`) on the controller HTTP server (requires `--tinker-backend`) |
| `--tinker-api-key` | X-API-Key the frontend requires (prefer `$MILES_TINKER_API_KEY` — a CLI flag shows in the process list); mandatory for a non-loopback bind |

The operator plane (`/adapter_runs*`, `/info`) accepts loopback peers only,
whatever the bind: the SDK key is a client credential and never grants the
routes that read server-local YAML files, choose save paths, or deregister
tenants. `/health` is liveness (the socket is up); `/api/v1/healthz` is
readiness and answers 503 until the driver reports the trainer exists.

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
poll `get_operation`, then `ack_operation` to release the record. These verbs
are the controller actor's Ray API; the tinker frontend drives them over
HTTP. Backpressure raises a retryable `OperationBackpressure` — the frontend
maps it to 429 + Retry-After, never to a 4xx the SDK treats as fatal.
Deregistering fences every open operation of that registration as
`FAILED(user)`.

Gradient-window poison: `optim_step` delimits a window of `forward_backward`
operations. If any of them reached a terminal state without succeeding (a
rejected chunk, an execution failure, a cancel), the window holds PARTIAL
gradients — the window's `optim_step` executes as a discard (all ranks clear
the slot's gradient sum), terminal-fails `FAILED(user)`, and moves neither
the step clock nor the serving version. The consumed poison resets the
window; resubmit the batch and step again.

## Tinker SDK frontend (tinker==0.24.1 JSON subset)

With `--tinker-frontend` the controller's HTTP server also speaks the REST
protocol of the official [`tinker`](https://pypi.org/project/tinker/) SDK —
exactly the **`tinker==0.24.1` JSON core-loop subset** (wheel source and
captured traffic; pure JSON, no protobuf: `/api/v1/client/config` pins the
SDK to its own default JSON path). Other SDK versions are rejected at
bootstrap (`/client/config` and `create_session` fail fast on the reported
`sdk_version`): 0.25+ switches `forward_backward` to protobuf, and the
current cookbook's canonical final checkpoint needs named sampler
checkpoints — neither is served here, so this is NOT "current
Tinker/cookbook compatible". An unmodified 0.24.1 client drives training
and sampling:

```python
import tinker
sc = tinker.ServiceClient(base_url="http://127.0.0.1:8068", api_key="tml-...")
tc = sc.create_lora_training_client(base_model=..., rank=32)
tc.forward_backward(data, "cross_entropy")
tc.optim_step(tinker.types.AdamParams(learning_rate=1e-4)).result()
sampler = tc.save_weights_and_get_sampling_client()
future = sampler.sample(                   # sample()/sample_async() submit /api/v1/asample;
    prompt=tinker.types.ModelInput.from_ints(prompt_tokens),
    num_samples=4,
    sampling_params=tinker.types.SamplingParams(max_tokens=128, temperature=0.7),
)
response = future.result()                 # .sequences[i].tokens / .logprobs / .stop_reason
```

Mapping: one training client = one registration (`create_model` registers,
`unload_model` deregisters), and every operation is pinned to its
`(name, registration_id)` — a stale handle fences instead of binding to a
same-name successor; every training verb forwards its SDK `seq_id` as the
registration ordinal (chunks posted out of order gap-buffer); futures poll
`/api/v1/retrieve_future` and terminal bodies replay until delivered (an
evicted delivered result leaves a fingerprint tombstone that answers a typed
410 — the 0.24.1 SDK surfaces it as a retryable "promise expired", it does
not re-run the original request); `save_state` mints `tinker://` paths
(resolved from an in-memory catalog; failures echo the public URI, not the
trainer filesystem); the ephemeral `save_weights_and_get_sampling_client`
publish binds `(name, registration_id, serving_version)` and samples through
the sglang router — a republish makes older sampling clients fail loud, and
the version is re-checked after generation so a publish landing mid-flight
fails the in-flight sample instead of returning cross-version output (the
identity is versioned, not leased: a publish committing between that check
and delivery is a documented residual race). Frontend rejections on a spent
`seq_id` become terminal `FAILED(user)` futures so the ordinal is still
consumed — bounded by the same unacked-results budget as every other record
(429 past it).

Frontend-level v1 rejections (beyond the backend matrix): non-0.24.x SDK
versions, LoRA `seed` and per-module `train_*` flags (deployment-wide),
weights-only restore (`load_state` / `create_training_client_from_state` —
the backend restores the full training state; use the `_with_optimizer`
variants), named persistent sampler checkpoints
(`save_weights_for_sampler(name)` / `create_sampling_client(model_path=...)`),
`ttl_seconds` (checkpoint/sampler TTL expiry is not implemented),
`topk_prompt_logprobs`, sparse-CSR tensors, and negative
token ids anywhere (targets, inputs, prompts, stop tokens). A sampling
`seed` maps to sglang `sampling_seed`, offset per sample so
`num_samples > 1` stays diverse. `prompt_logprobs` maps to sglang
`logprob_start_len=0` on the same generate (the engine scores the prompt
natively; position 0 has no context and returns null) — this serves both
`sample(include_prompt_logprobs=True)` and the SDK's `compute_logprobs()`,
which the 0.24.1 wheel sends as a 1-sample, 1-token generation.

Sampling architecture: `/asample` returns its future immediately and a
background task posts one router `/generate` per sample, carrying the
server-derived serving identity (`rid`/`lora_path`/`extra_key` are never
client-controllable — the wire models drop unknown fields and the sglang
params are rebuilt from an allowlist). SGLang's continuous batching is the
only sampling batcher: the frontend never coalesces prompts, and the
training-operation scheduler (`MultiLoraOperationBatchFn`) never sees a sampling
request. The legacy datasource rollout pipeline
(`RolloutManager.generate()`: datasets, rewards, training-data conversion)
is not on this path — the frontend shares only the router the rollout
engines already serve.

Trust boundary (v1): the frontend authenticates clients and bounds aggregate
active sub-generations, rejects one request whose `num_samples` exceeds that
capacity, and preflights `prompt + max_tokens` against the discovered engine
limit. It still does not validate token ids against the vocabulary upper bound
or enforce request-body/output-byte quotas. Run it loopback/VPN-facing for
trusted clients; per-tenant quotas are future work.

## v1 compatibility matrix

Supported: text-only input; the synchronous training loop; 1-D shifted
targets; `loss_fn ∈ {cross_entropy, importance_sampling, ppo}` (per-op clip
config); per-call AdamParams; multi-chunk gradient accumulation with
independent `optim_step`; latest-only sampler weights behind the publish
barrier; prompt logprobs (`compute_logprobs()` /
`sample(include_prompt_logprobs=True)`, one sub-generation of admission
weight); named immutable `save_state` / `load_state` (create-from-checkpoint
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
