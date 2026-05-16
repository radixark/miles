# Multi-LoRA Training Example

Train multiple LoRA adapters concurrently against a shared base model. Each
adapter gets its own dataset, reward type, hyperparameters, and step counter,
but rollouts are interleaved on the same SGLang engines via slot-based hot
swapping.

This example trains two adapters on Qwen3-4B:

- **gsm8k** — grade-school math, `rm_type: math`
- **dapo_math** — competition math (DAPO-Math-17k), `rm_type: deepscaler`

## Layout

```
provision.sh                    # one-time: download model + datasets
single_run.sh                   # entrypoint: bounded run, exits when done
start_service.sh                # entrypoint: service mode, waits for adapters
submit_schedule.sh              # companion: drives register/deregister
run_schedule.py                 # schedule script invoked by submit_schedule.sh
train_multi_lora.py             # trainer (used by both entrypoints)
adapters/
  gsm8k/adapter.yaml
  dapo_math/adapter.yaml
```

## Provision (once)

```bash
bash examples/multi_lora/provision.sh
```

Downloads `Qwen/Qwen3-4B`, `zhuzilin/dapo-math-17k`, and `zhuzilin/gsm8k`.

## Entrypoints

### `single_run.sh` — bounded run

Registers the two adapters from CLI flags and trains until each adapter hits
its `num_row` (or `--num-rollout`), then exits.

```bash
bash examples/multi_lora/single_run.sh
```

Key behavior:

- Adapters are passed via `--multi-lora-adapter <name> <adapter.yaml path>`
  (repeatable).
- `--multi-lora-disable-service-mode` makes the trainer exit once all
  registered adapters have drained.

### `start_service.sh` — long-running service

Starts the trainer with **no** adapters and leaves it idle, polling for
registrations. You then push adapters/schedules from another process.

```bash
# Terminal 1
bash examples/multi_lora/start_service.sh

# Terminal 2 (after Ray dashboard is up)
bash examples/multi_lora/submit_schedule.sh
```

`submit_schedule.sh` runs `run_schedule.py`, which talks to the running
trainer's controller actor and fires register/deregister events on a
predefined timeline (see the docstring in `run_schedule.py`).

## Multi-LoRA CLI flags

These are added on top of the standard LoRA flags (`--lora-rank`,
`--lora-alpha`, `--target-modules`, ...):

| Flag | Purpose |
| --- | --- |
| `--multi-lora-n-adapters N` | Max concurrent adapter slots. `0` disables multi-LoRA (default `0`); any value `> 0` enables it. |
| `--multi-lora-adapter NAME PATH` | Register an adapter at startup. Repeatable. `PATH` points at an `adapter.yaml`. |
| `--multi-lora-idle-poll-s SECONDS` | How often the trainer polls for new adapters when idle (default `5.0`). |
| `--multi-lora-disable-service-mode` | Exit once all registered adapters have drained. Default is to wait forever. |

In multi-LoRA training, the `--lora-rank` arg defines the **upper bound** on rank across all slots; per-adapter
`rank` in `adapter.yaml` must be `<= --lora-rank`.

## adapter.yaml

```yaml
rank: 16                                       # <= adapter lora rank
alpha: 16
data: /root/gsm8k/train.parquet                # parquet/jsonl path
input_key: messages                            # prompt column
label_key: label                               # ground-truth column
rm_type: math                                  # reward function
num_row: 400                                   # stop adapter after N rows
# optional: dir, num_epoch, custom_rm_path, ...
```

`dir` defaults to the directory containing `adapter.yaml`; the trainer writes
checkpoints there.
