# Multi-LoRA Training Example (fully-async)

Train multiple LoRA adapters concurrently against a shared base model, using a
fully-async rollout (continuous producer) + a slot-keyed LoRA page table on the
SGLang engines (in-place upsert, no unload, no drain).

This example trains two adapters on Qwen3-4B:

- **gsm8k** — grade-school math, `rm_type: math`
- **dapo_math** — competition math (DAPO-Math-17k), `rm_type: deepscaler`

## Layout

```
run_multi_lora.py                    # launcher: prepare / train / full-train / serve
service_smoke.py                     # register/deregister smoke test against the API
adapters/
  gsm8k.yaml
  dapo_math.yaml
```

The implementation lives in the library: the driver is `train_multi_lora_async.py`
at the repo root (next to `train.py`/`train_async.py`), the rollout fn and data
source are `miles/rollout/multi_lora/`, and the controller is
`miles/ray/multi_lora/` (registry + backend + HTTP API, plus the named Ray
actor pinned to the head node).

## Design (decoupled per-adapter optimizers)

- **Controller** (Ray actor + control-plane HTTP API) is the source of truth:
  `POST/GET/DELETE /adapter_runs` plus `GET /adapter_runs/state`. The data source
  reads it; the trainer reads it. Generation traffic goes straight to the router;
  on deregister the controller aborts the adapter's in-flight requests
  engine-side by rid prefix (`rid = {adapter}::{uuid}`, set in `generate`).
- **Per-adapter gradient accumulation.** Each adapter has its own batch shape:
  `rollout_batch_size` prompt groups per optimizer step, each group holding
  `n_samples_per_prompt` responses (`adapter_global_batch_size =
  rollout_batch_size x n_samples_per_prompt` samples per step). Completed
  prompt groups flow into training continuously in multiples of the
  adapter's `min_groups_per_dp_split` (the smallest group count whose samples
  split evenly across data-parallel ranks), gradients
  accumulate in the DDP buffers across train batches, and an adapter's
  optimizer steps exactly when its adapter batch fills — independent of every other
  adapter. The controller tracks adapter batch progress (`accumulated_groups`) and commits
  it only after a successful train call.
- **Per-slot optimizers.** One Adam per adapter slot under Megatron's
  `LayerWiseDistributedOptimizer` (whole-parameter ZeRO-1): per-slot state,
  step counts, and gradient clipping; optimizer state sharded across DP ranks;
  plain DDP all-reduce (no distributed optimizer) makes cross-batch gradient
  retention idempotent.
- **Batch collection.** The collection loop (same shape as fully_async's)
  pops groups from the per-adapter buffers round-robin, one
  `min_groups_per_dp_split` at a time, capped at each adapter's remaining
  batch, until the batch reaches `--global-batch-size` samples or a non-empty
  batch makes no progress for `--multi-lora-max-coalesce-wait-s` (the target
  can be permanently unreachable, so it trains on whatever is ready) — a
  single adapter with a small batch trains alone without waiting for
  anyone. Samples enter the gradient buffers with weight 1; at step time the
  slot's accumulated gradient is scaled by `1/adapter_global_batch_size`
  (a constant known in advance), so an adapter's update is identical to what
  it would get training alone.
- **Selective weight sync.** Only adapters whose optimizer stepped are pushed
  to the engines (upsert into the slot-keyed page table); only their slot
  versions bump, keeping staleness filtering per-adapter accurate.
- Adapters deregister on committed optimizer-step count (`num_step`) in the
  controller's train-commit path (`mark_batch_trained`), so stop checks happen
  exactly when steps advance. `num_step` is relative to the adapter's
  start/resume step. When an adapter doesn't set `num_step`, it is derived
  from `num_epoch` (default 1) as `num_epoch x len(dataset) //
  rollout_batch_size` once the data source loads the dataset (post-filter
  length). The trainer's
  `reconcile_adapters` (before each generate) retires it at the next sync
  point and cleans up (save ckpt + clear Megatron slot + zero its optimizer
  state and retained gradients). The adapter's untrained tail — buffered
  groups and any partially accumulated gradients — is discarded.
- **Batch ⊆ loaded property:** `reconcile_adapters` runs before `generate`, so the
  batch is fetched with loaded = active; active only shrinks during generate, so every
  adapter in the batch is live on the trainer.

## Provision (once)

```bash
python examples/multi_lora/run_multi_lora.py prepare
```

Downloads `Qwen/Qwen3-4B` (to `/root/models`), `zhuzilin/dapo-math-17k`, and
`zhuzilin/gsm8k` (to `/root/datasets`).

## Run

```bash
python examples/multi_lora/run_multi_lora.py train        # or: full-train (prepare + train)
```

Registers the two adapters from CLI flags and trains until each hits its `num_step`,
then exits.

## Service mode

```bash
python examples/multi_lora/run_multi_lora.py serve
```

Starts with no adapters and idles; register/deregister at runtime through the
control-plane API (port 8068):

```bash
python examples/multi_lora/service_smoke.py --api-url http://127.0.0.1:8068 \
    --data /root/datasets/gsm8k/train.parquet --input-key messages --label-key label --rm-type math
```

## Multi-LoRA CLI flags

| Flag | Purpose |
| --- | --- |
| `--multi-lora-n-adapters N` | Max concurrent adapter slots. `0` disables (default); `> 0` enables. |
| `--multi-lora-adapter NAME PATH` | Register an adapter at startup. Repeatable. `PATH` → an `adapter.yaml`. |

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
num_step: 400               # stop adapter after N optimizer steps
                            # (default: derived from num_epoch, itself default 1)
# optional: save, num_epoch, custom_rm_path, ...
```

The derived `adapter_global_batch_size = rollout_batch_size x
n_samples_per_prompt` is the adapter's samples-per-optimizer-step (the
per-adapter analog of `--global-batch-size`).

Batch-shape constraints (validated at registration, not at runtime):
`n_samples_per_prompt` must be a divisor or multiple of the trainer's
data-parallel size; `rollout_batch_size` must be a multiple of the adapter's
`min_groups_per_dp_split`;
`adapter_global_batch_size` is capped by
`--multi-lora-max-adapter-global-batch-size` (default 4x `--global-batch-size`).
