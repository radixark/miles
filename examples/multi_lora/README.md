# Multi-LoRA Training Service

Train many LoRA adapters concurrently against one shared base model: one
Megatron trainer holds a pool of adapter slots, shared SGLang engines serve
every adapter's rollouts, and a control-plane HTTP API registers and retires
training runs at runtime — adapters may outnumber slots.

Each registered adapter is an **independent run**: its own dataset, reward,
batch shape, learning-rate clock, stop condition, and checkpoints. This
example trains two adapters on Qwen3-4B:

- **gsm8k** — grade-school math, `rm_type: math`
- **dapo_math** — competition math (DAPO-Math-17k), `rm_type: deepscaler`

## Layout

```
run_multi_lora.py            # launcher: prepare / train / full-train / serve
register_and_train.py        # client example: register runs via the API, watch them finish
adapters/
  gsm8k.yaml
  dapo_math.yaml
```

The implementation lives in the library: the driver is
`train_multi_lora_async.py` at the repo root, the rollout frontend is
`miles/rollout/multi_lora/` (`MultiLoRARolloutFn`), and the controller is
`miles/ray/multi_lora/` (registry + slot pool + HTTP API on a named Ray actor).

## How it works

- **Controller** (Ray actor + HTTP API) is the source of truth:
  `POST/GET/DELETE /adapter_runs`. Serving identity is registration-scoped —
  a re-registered name is a new tenant, so stale in-flight requests and KV
  cache entries from a previous registration can never leak into the new run.
- **Whole batches, atomically.** Each adapter's child rollout produces one
  complete logical batch (`rollout_batch_size` prompt groups ×
  `n_samples_per_prompt` responses). A persistent round-robin selection
  coalesces ready batches toward `--global-batch-size` (waiting at most
  `--multi-lora-max-coalesce-wait-s`), and a selected batch always ships
  whole — an adapter's optimizer step sees exactly the batch shape its yaml
  declared, independent of every other adapter.
- **Per-slot optimizers and schedulers.** One Adam per slot under Megatron's
  `LayerWiseDistributedOptimizer`; per-slot learning-rate schedulers tick on
  that adapter's optimizer steps. A non-finite gradient vetoes only the
  offending adapter's step — its clocks don't advance and nothing publishes.
- **Slot oversubscription.** Registrations beyond the slot pool queue unbound
  and bind when a slot frees (bootstrap) or at selection time via
  transactional reservations; idle residents are evicted keep-warm (LRU) with
  optimizer-inclusive sidecar checkpoints, so a swapped-out adapter resumes
  bit-exact — weights, fp32 masters, Adam moments, and step counters.
- **Publish gate.** Only adapters whose step committed push weights, under
  their registration-scoped serving name; an adapter's next batch starts only
  after its previous step's weights are live on the engines.
- **Lifecycle.** A run retires automatically once committed steps reach
  `num_step` (derived from `num_epoch`, default 1, when unset): final
  checkpoint saved, slot cleared, in-flight requests aborted by rid prefix.
  Re-registering the same name resumes from its saved checkpoint.

## Provision (once)

```bash
python examples/multi_lora/run_multi_lora.py prepare
```

Downloads `Qwen/Qwen3-4B` (to `/root/models`), `zhuzilin/dapo-math-17k`, and
`zhuzilin/gsm8k` (to `/root/datasets`).

## Bounded run

```bash
python examples/multi_lora/run_multi_lora.py train        # or: full-train (prepare + train)
```

Registers the two adapters from `adapters/` at startup and exits once each
reaches its `num_step`.

## Service mode

```bash
python examples/multi_lora/run_multi_lora.py serve
```

Starts with no adapters and idles; register and watch runs from any machine
that can reach the API (port 8068):

```bash
python examples/multi_lora/register_and_train.py \
    --api-url http://127.0.0.1:8068 \
    --adapter gsm8k=examples/multi_lora/adapters/gsm8k.yaml \
    --adapter dapo_math=examples/multi_lora/adapters/dapo_math.yaml
```

`tests/manual/multi_lora_service_smoke.py` exercises the full register/train/deregister lifecycle
(including mid-run registration, mid-run deregistration, and name reuse) and
is what the GPU E2E scripts assert against.

## Multi-LoRA CLI flags

| Flag | Purpose |
| --- | --- |
| `--multi-lora-n-adapters N` | Adapter slot pool size. `0` disables (default); registrations beyond `N` queue unbound. |
| `--multi-lora-adapter NAME PATH` | Register an adapter at startup. Repeatable. `PATH` → an `adapter.yaml`. |
| `--multi-lora-api-port PORT` | Control-plane API port on the head node (default 8068). |
| `--multi-lora-disable-service-mode` | Exit after all startup adapters finish instead of idling for registrations. |
| `--multi-lora-idle-poll-s S` | Poll cadence for new registrations while no adapter is active. |
| `--multi-lora-max-coalesce-wait-s S` | How long ready batches wait to coalesce toward `--global-batch-size`. |
| `--multi-lora-max-empty-wait-s S` | How long a generate call waits for the first ready batch. |

Per-adapter `rank` in `adapter.yaml` must be `<= --lora-rank`.

## adapter.yaml

```yaml
rank: 16
alpha: 16
rollout_batch_size: 32      # prompt groups per optimizer step (defaults to --rollout-batch-size)
n_samples_per_prompt: 4     # group shape (defaults to --n-samples-per-prompt)
data: /root/datasets/gsm8k/train.parquet
input_key: messages
label_key: label
rm_type: math
num_step: 400               # stop after N committed optimizer steps
                            # (default: derived from num_epoch, itself default 1)
# optional: save, num_epoch, custom_rm_path, metadata, rollout_function_path
```

## Scheduling semantics and boundaries

- **Oversubscription is queue-first.** A registration beyond the slot pool
  waits unbound and binds when a slot frees (an earlier run retiring); it does
  not evict a resident adapter at selection time. The transactional
  bind-at-selection machinery exists, but reaching it requires the rollout
  engines to serve more adapters than the trainer has slots — today
  `--multi-lora-n-adapters` sizes both, so eviction stays dormant.
- **Batch shape.** A selection ships whole per-adapter batches and trains them
  as one step; nothing is trimmed and no dp-divisibility is required. The step
  must still be splittable into `dp_size` micro-batches (with pipeline
  parallelism disabled that is the only alignment): a selection whose sample
  count is below `dp_size`, or whose samples individually exceed
  `--max-tokens-per-gpu` in a way that leaves fewer splittable micro-batches
  than ranks, fails the schedule with an explicit assertion.
- **Agentic children.** One rollout execution may emit several sibling samples
  (shared `rollout_id`); the loss and the per-adapter step normalization both
  count executions, not samples. Size agentic runs with the per-adapter
  `rollout_batch_size` — the per-execution trajectory count multiplies the
  physical batch and is invisible to registration-time validation.
- On MoE models the grouped-GEMM adapter path supports up to
  `1024 / experts_per_rank` concurrent slots (a `torch._grouped_mm` limit;
  higher expert parallelism raises the ceiling).
