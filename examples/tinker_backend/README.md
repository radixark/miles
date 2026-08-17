# Tinker-compatible backend

Serve many LoRA training runs on one shared base model through a
[tinker](https://tinker-docs.thinkingmachines.ai/)-style operation API: clients
drive training with explicit `forward_backward` / `optim_step` operations and
sample through the shared engines — no dataset, no reward function, and no
batch schedule on the server.

```
official tinker SDK ──HTTP──> TinkerController (head node)
                                ├─ tinker frontend      /api/v1 (--tinker-frontend; the REST
                                │                       protocol tinker==0.24.1 speaks)
                                ├─ registration plane   /adapter_runs (operator surface)
                                ├─ operation ledger     enqueue → claim → complete → ack
                                └─ serving plane        sglang router (sampling proxied)
trainer ranks <──Ray── driver loop (train_tinker_backend.py)
```

## Start the Miles engine

For the documented SDK flow, start both the operation backend and the Tinker
frontend. The helper starts the shared training and sampling engines in
service mode; add `--tinker-frontend` through `--extra-args` so that the
official SDK can use the controller's `/api/v1` endpoint:

```bash
# Once per node: download the example checkpoint.
python examples/tinker_backend/run_tinker_backend.py prepare

# Start Miles in service mode, with both the backend and frontend enabled.
python examples/tinker_backend/run_tinker_backend.py serve \
  --extra-args "--tinker-frontend"
```

The following lower-level command is useful when deploying with custom
Megatron and SGLang flags:

```bash
python train_tinker_backend.py \
  --tinker-backend \
  --tinker-frontend \
  --multi-lora-n-adapters 4 \
  --lora-rank 32 --lora-alpha 64 \
  --target-modules all-linear \
  --hf-checkpoint Qwen/Qwen3-0.6B \
  ... # the usual megatron/sglang flags; see run_tinker_backend.py
```

Key flags:

| flag | meaning |
|------|---------|
| `--tinker-backend` | enable the operation backend (requires `--multi-lora-n-adapters > 0`) |
| `--multi-lora-n-adapters N` | fixed slot count; a registration binds a slot for life (queue when full) |
| `--lora-rank` / `--lora-alpha` | deployment-wide ceiling / fixed alpha — clients may lower `rank`, never set `alpha` |
| `--multi-lora-disable-service-mode` | exit once all adapters retire (by default the service keeps serving with zero adapters) |
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
`--recompute-modules` when expert modules are targeted — additionally
requires a Megatron-Bridge whose PEFT recompute patch recognizes multi-LoRA
`.adapters.<slot>.` params (radixark/Megatron-Bridge#27, branch `bridge` @
`688d34b8`): multi-LoRA trains adapter-only, so those checkpointed regions
replay grad-enabled only because that patch forces TransformerBlock inputs to
require grad. Launch probes the installed bridge and refuses the two shapes
on an unfixed one, where every adapter gradient is silently zero and the job
steps forever at `grad_norm=0.0` without learning (4xH200 GPT-OSS 20B repro,
2026-08-12; full recompute re-validated training real gradients on the fixed
bridge, same config).

## Operation contract

`enqueue_operation(name, operation_id, ordinal, kind, payload)` — ordinals are
consecutive per registration starting at 1; arrival may be out of order
(gap-buffered, and a hole-filling ordinal is always admitted), execution is
strictly ordinal-ordered; retries with the same `operation_id`, same ordinal,
and identical payload return the original operation — anything else is a
typed conflict.

| kind | payload | success result |
|------|---------|----------------|
| `forward_backward` | `{samples: [Datum...], loss: {loss_fn, loss_fn_config?}}` | `{logprobs: [[...]], metrics: {"loss:sum", "unmasked_tokens:sum"}}` |
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

### Client-owned RL loop

After the engine reports ready, connect the official SDK client to the
frontend endpoint and run the loop below. The backend executes each requested
operation; rollout generation, scoring, and `Datum` construction remain in
the client.

Start the driver with both `--tinker-backend` and `--tinker-frontend`.  The
backend then owns execution and serving, while the client owns data
preparation and the training loop.  In particular, the client can run the
same pattern as the [target-flow example](https://github.com/radixark/miles/issues/2258):

```python
import tinker
from transformers import AutoTokenizer

service = tinker.ServiceClient(base_url="http://127.0.0.1:8068", api_key="tml-...")
base_model = service.get_server_capabilities().supported_models[0].model_name
training = service.create_lora_training_client(base_model=base_model, rank=16)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Publish the initial LoRA so the first rollout has a policy to sample.
sampler = training.save_weights_and_get_sampling_client()

rl_prompts = ["Solve: If a train travels 60 km in 2 hours, what is its speed?"]
prompt_ids = [tokenizer(p).input_ids for p in rl_prompts]

for update_idx in range(num_rl_updates):
    # Option 1 -- SFT data preparation (client-owned; replace the RL batch
    # below and train with loss_fn="cross_entropy").
    # batch = [
    #     datum_from_sft_example(example["prompt"], example["completion"])
    #     for example in sft_examples
    # ]

    # Option 2 -- RL data preparation (client-owned). sample() returns a
    # future; .result() carries sequences with tokens and logprobs.
    futures = [
        sampler.sample(
            prompt=tinker.types.ModelInput.from_ints(ids),
            num_samples=4,
            sampling_params=tinker.types.SamplingParams(max_tokens=256, temperature=1.0),
        )
        for ids in prompt_ids
    ]
    rollouts = [future.result() for future in futures]
    scored = score_rollouts(rl_prompts, rollouts)  # rewards -> advantages, client-owned
    batch = [
        datum_from_scored_rollout(ids, sequence, advantage)
        for ids, response, advantages in zip(prompt_ids, rollouts, scored)
        for sequence, advantage in zip(response.sequences, advantages)
    ]

    fb = training.forward_backward(batch, "importance_sampling")
    step = training.optim_step(tinker.types.AdamParams(learning_rate=1e-4))
    fb.result()
    step.result()

    # Publish explicitly so the next rollout samples the new policy.
    # Serving is latest-only: the publish supersedes the previous sampling
    # client, so re-acquire it here every update.
    sampler = training.save_weights_and_get_sampling_client()
```

`datum_from_sft_example`, `score_rollouts`, and `datum_from_scored_rollout`
are application code: they define the task data, rollout scoring, and the
per-token loss channels. An RL datum pairs `model_input` (prompt + sampled
tokens, shifted) with `loss_fn_inputs` `target_tokens`, the sampler's
returned `logprobs`, and per-token `advantages`; an SFT datum needs
`target_tokens` plus 0/1 `weights`. The frontend translates the resulting
SDK requests to operations; the backend executes them in order and only
changes the sampler's policy on the explicit publish. The complete runnable
version of this loop is `tests/e2e/tinker_backend/tinker_sdk_rl_quality.py`
(GRPO on GSM8K, four concurrent adapters through one deployment).

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
`ttl_seconds` (no reaper runs; a recorded TTL would be a false promise),
`prompt_logprobs` / `topk_prompt_logprobs`, sparse-CSR tensors, and negative
token ids anywhere (targets, inputs, prompts, stop tokens). A sampling
`seed` maps to sglang `sampling_seed`, offset per sample so
`num_samples > 1` stays diverse.

Sampling architecture: `/asample` returns its future immediately and a
background task posts one router `/generate` per sample, carrying the
server-derived serving identity (`rid`/`lora_path`/`extra_key` are never
client-controllable — the wire models drop unknown fields and the sglang
params are rebuilt from an allowlist). SGLang's continuous batching is the
only sampling batcher: the frontend never coalesces prompts, and the
training-operation scheduler (`TinkerRolloutFn`) never sees a sampling
request. The legacy datasource rollout pipeline
(`RolloutManager.generate()`: datasets, rewards, training-data conversion)
is not on this path — the frontend shares only the router the rollout
engines already serve.

Trust boundary (v1): the frontend authenticates clients but does not meter
them — token ids are not checked against the vocabulary (upper bound), and
request/fan-out/output quotas (`num_samples`, `max_tokens`, body bytes) are
not enforced. Run it loopback/VPN-facing for trusted clients; per-tenant
quotas are future work.

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

## Files

- `run_tinker_backend.py` — disaggregated launch (`prepare` / `serve` / `train`)
- `adapters/example.yaml` — CLI pre-registration example (`--multi-lora-adapter example adapters/example.yaml`)
