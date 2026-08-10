# GLM-5.2 744B-A40B — agentic RL on terminal-bench-2 (Daytona sandboxes)

Fully-async RL on 16 GB300 nodes (4 GPUs each): **8 training nodes**
(TP2/CP4/PP4/EP8, optimizer state streamed to node-local disk) and **8
inference nodes** (one 4-GPU dp-attention fp8 sglang engine per node). Every
episode is a multi-turn terminal agent solving one terminal-bench-2 task inside
its **own** Daytona cloud sandbox, built from that task's official image;
scoring is the task's canonical `tests/test.sh`.

All experiment settings live in `run_glm5_2_744b_a40b_daytona.py` — its
defaults are the reference configuration, and

```bash
python3 run_glm5_2_744b_a40b_daytona.py train --num-nodes 16
```

reproduces it. `launch_16node_slurm.sh` is a ~60-line site adapter (container +
Ray bring-up) that forwards its own CLI args to the recipe.

Reference run: 100 rollout steps in 21 h wall clock (~6.5 min/step incl. evals
every 10), rollout truncation 0.0–0.3, prefix-cache hit rate ≈0.96, engine
fleet fully utilized (~90–100 concurrent generating requests across 32 dp
ranks at `--async-max-concurrent-samples 128`).

## What the config does

| | |
|---|---|
| Training | 8 nodes, TP2 / CP4 / PP4 (18-20-20-20 layer split) / EP8. CP4 splits the 131k max sequence to ~33k per rank; DP1 leaves optimizer state unsharded, so `--stream-optimizer-state-to-disk` is mandatory, not an optimization |
| Inference | 8 nodes, one 4-GPU engine each: dp-attention (dp=4) + deepep, EAGLE 1/1/2, fp8 KV. `--sglang-config low-latency` switches to one TP8 engine per node pair with EAGLE 5/1/6 |
| KV budget | `--sglang-mem-fraction-static 0.85` → ~553k KV tokens per dp rank. At the older 0.75 the pool was 26k tokens: trajectories hard-failed past ~26k input, 84% of samples truncated, and one long trajectory saturated a rank |
| Async | `train_async.py` + `FullyAsyncRolloutFn`; 128 in-flight trajectories decoupled from the 64-sample train batch; `--rollout-submission-granularity sample` frees a submission slot per finished sample rather than per finished group |
| Routing | dp-attention implies dp-aware routing (set automatically); without it, requests pile onto one dp rank per engine while the other three idle |
| Episodes | 30 turns, 3600 s wall clock, 16k tokens/turn, 131k session; thinking at GLM-5.2's default `Reasoning Effort: Max` |
| Eval | every 10 rollouts over a held-out tbench2 split, on the shared rollout engines (the producer pauses). 20 tasks x 2 samples has σ≈0.08 — single-eval movements below that are noise; raise `--n-samples-per-eval-prompt` to tighten |
| Sandboxes | one per episode, deleted at episode end; labelled `openenv-tbench2-task` / `openenv-launcher` / `openenv-run-id` |

Rollout mechanics are shared with other recipes and live one directory up: the
agent loop and Daytona backend in `../openenv_daytona_agent_function.py`, the
reward hook in `../openenv_generate.py`, dataset generation in
`../make_tbench2_data.py`.

## Prerequisites

**Container image.** A miles runtime image with sglang from `sglang-miles`
at 2026-08-04 or later.

**Python packages**, installed in the image or once per node (the recipe
asserts them at launch):

```bash
pip install openenv daytona fastmcp
pip install --no-build-isolation --no-deps -e $OPENENV_ROOT/envs/tbench2_env
```

`tbench2_env` must be installed **editable from the OpenEnv checkout**, not
from a wheel: the per-task sandbox image recipe embeds the package source
(`pyproject.toml`, `openenv.yaml`) in the Daytona build context, and a wheel
install into `site-packages` carries neither.

**Megatron** with the streamed-optimizer checkpoint hooks
(radixark/Megatron-LM#63); pass the checkout via `--megatron-path`.

**terminal-bench-2 checkout** — task definitions; each `task.toml` names the
official `docker_image`. Point `OPENENV_TB2_TASKS_DIR` at it.

**Daytona** org with quota for `--async-max-concurrent-samples` concurrent
sandboxes (128 in the reference config; each 2 vCPU / 4 GiB / 10 GiB). Keep
the credential in a file outside git:

```bash
printf 'export DAYTONA_API_KEY=dtn_...\n' > ~/.daytona_env && chmod 600 ~/.daytona_env
```

Images are built per task and cached by definition hash (first episode of a
task ~10 min, repeats ~1 min).

**Model and data** under `--model-dir` / `--data-dir`:

```bash
$MODEL_DIR/GLM-5.2_torch_dist   # training; tools/convert_hf_to_torch_dist.py
$MODEL_DIR/GLM-5.2_fp8          # rollout;  tools/convert_hf_to_fp8.py

python examples/experimental/openenv/make_tbench2_data.py \
    --tasks_dir $OPENENV_TB2_TASKS_DIR --output $DATA_DIR/tbench2_train.jsonl
# hold out a disjoint task set as $DATA_DIR/tbench2_eval.jsonl for the eval
```

Prompt `metadata.task_id` must match a directory name in the TB2 checkout —
that id selects the per-episode sandbox image.

## Run

```bash
export MILES_ROOT=...  CONTAINER_IMAGE=...  CONTAINER_MOUNTS=...
export DAYTONA_ENV_FILE=~/.daytona_env
export OPENENV_TB2_TASKS_DIR=...
export WANDB_API_KEY=...            # optional

sbatch --export=ALL examples/experimental/openenv/glm52_tbench2/launch_16node_slurm.sh \
    --model-dir $MODEL_DIR --data-dir $DATA_DIR --output-dir $OUTPUT_DIR
```

Anything after the launcher name goes to the recipe: `--num-rollout 10` for a
short run, `--sglang-config low-latency`, `--eval-interval` to change cadence.
Outside slurm, bring up your own Ray cluster and run the recipe command from
the module docstring on the head node.

Weights load from `--model-local-dir` (default: same as `--model-dir`);
pre-staging both checkpoints to node-local disk saves a multi-minute NFS read
per engine restart.

## Watching it

Telemetry lands under `$OUTPUT_DIR/<run_id>/dump_details`; serve the dashboard
with `python -m miles.dashboard.serve --dump-details <dir> --follow`. Useful
driver-log signals:

```bash
grep -E "raw_reward|truncated_ratio" run.out      # learning signal
grep -c "episode failed" run.out                   # sandbox/env health (a few %% is normal)
```

Healthy steady state: `rollout/truncated_ratio` mostly under 0.3 (it tracks the
hard-task mix), `queue_size` near 0, episode failures a few percent (Daytona
connection blips; backfill absorbs them).

## Debugging aids built into the recipe

- `--debug-replay-data '<dir>/{rollout_id}.pt'` — replay recorded rollout
  batches through the training side only (8 nodes, no engines, no sandboxes).
  This is how parallelism/OOM changes are validated in minutes: the dumps under
  `dump_details/rollout_data/` are loadable as-is.
- `--load-from <run>/checkpoints` plus `--extra-args '--start-rollout-id 0'` —
  score an existing checkpoint: eval runs on the loaded weights before any
  training step. Keep `--num-rollout` equal to the source run's (the LR
  scheduler validates it).
