# Multi-LoRA Training Example (fully-async)

Train multiple LoRA adapters concurrently against a shared base model, using a
fully-async rollout (continuous producer) + a slot-keyed LoRA page table on the
SGLang engines (in-place upsert, no unload, no drain).

This example trains two adapters on Qwen3-4B:

- **gsm8k** — grade-school math, `rm_type: math`
- **dapo_math** — competition math (DAPO-Math-17k), `rm_type: deepscaler`

## Layout

```
provision.sh                         # one-time: download model + datasets
run_job.sh                           # entrypoint: bounded run, exits when done
run_service.sh                       # service mode: idles for registrations (port 8068)
service_smoke.py                     # register/deregister smoke test against the API
train_multi_lora_async.py            # trainer (entry point)
multi_lora_async_rollout.py          # fully-async rollout function
multi_lora_data_source_async.py      # data source (reads controller, deregisters at num_row)
adapters/
  gsm8k.yaml
  dapo_math.yaml
```

Controller code lives in the library: `miles/utils/multi_lora.py` (registry +
backend + HTTP API, torch-free) and `miles/ray/multi_lora_controller.py` (named
Ray actor, pinned to the head node).

## Design (no drain, no state machine)

- **Controller** (Ray actor + control-plane HTTP API) is the source of truth:
  `POST/GET/DELETE /adapter_runs` plus `GET /adapter_runs/state`. The data source
  reads it; the trainer reads it. Generation traffic goes straight to the router;
  on deregister the controller aborts the adapter's in-flight requests
  engine-side by rid prefix (`rid = {adapter}::{uuid}`, set in `generate`).
- **No drain / no rollout-id / no train_steps / no PENDING-DRAINING-DRAINED states.**
  The data source deregisters an adapter at `num_row`; the trainer's
  `reconcile_adapters` (before each generate) cleans up gone adapters (save ckpt +
  clear Megatron slot) and loads new ones. `update_weights` upserts active adapters'
  weights in place (SGLang page table, `upsert=True`).
- **Batch ⊆ loaded property:** `reconcile_adapters` runs before `generate`, so the
  batch is fetched with loaded = active; active only shrinks during generate, so every
  adapter in the batch is live on the trainer.

## Provision (once)

```bash
bash examples/multi_lora/provision.sh
```

Downloads `Qwen/Qwen3-4B`, `zhuzilin/dapo-math-17k`, and `zhuzilin/gsm8k`.

## Run

```bash
bash examples/multi_lora/run_job.sh
```

Registers the two adapters from CLI flags and trains until each hits its `num_row`
(or `--num-rollout`), then exits.

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
data: /root/gsm8k/train.parquet
input_key: messages
label_key: label
rm_type: math
num_row: 400                # stop adapter after N rows
# optional: save, num_epoch, custom_rm_path, ...
```
