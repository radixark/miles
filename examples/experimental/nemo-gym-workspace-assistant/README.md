# Nemotron Workplace Assistant

Train Nemotron 3.5 Lightning to use email, calendars, project boards, and CRM
tools in [NeMo Gym's Workplace Assistant](https://github.com/NVIDIA-NeMo/Gym/tree/1ea6b75496c97bf52cdc4578bf34afc4fc5e9e7a/resources_servers/workplace_assistant).
This is a simulated office: no real accounts, browser, Docker sandboxes, or
external workplace services are involved.

The agent makes several model/tool turns per task. Miles' session server records
the original tokens and log probabilities and masks tool observations out of the
training loss. A separate CPU service executes Gym's native tools in a fresh
environment for each episode. Gym awards **1 for the correct final state, 0
otherwise**; the model's final claim of success does not determine reward.
Reference actions stay in the resource service, outside the policy prompt.

This experimental recipe ports the settings from this
[reference run](https://wandb.ai/radixarkai/nemotron35-workplace-assistant/runs/kk6omvbp).
The recipe needs GPU validation with your installed Miles/SGLang/Megatron versions.

For a beginner-oriented walkthrough, including data-generation and calibration
commands, see [Train a workplace assistant with Miles](WALKTHROUGH.md).

## 1. Prepare the machines

Use two nodes with **8 H200 GPUs each**, a working Miles training installation,
and the same model, Miles, Megatron-LM, and dataset paths on both nodes.
Use the current Miles container/setup described in the
[quick start](../../../docs/getting-started/quick-start.md).
The trainer uses TP=2, PP=2, EP=2; the other node runs eight TP=1 rollout engines.
This uses the batch-overlapped `train_async.py` driver, not `--fully-async`.

Choose a large writable output disk. In the reference run each full optimizer
checkpoint was about 442 GB: keeping ten requires roughly 4.4 TB before traces
and model caches. Start with a new output directory to avoid mixing runs.

The following commands use **Fish**. Run in the Miles Python environment, with
`uv` installed. Set these paths to your own directories:

```fish
set -gx MILES_DIR /path/to/miles
set -gx GYM_DIR /path/to/Gym
set -gx WORKPLACE_DATA /data/workplace
set -gx MODEL_DIR /data/models
set -gx PYTHON (command -v python)
git clone https://github.com/NVIDIA-NeMo/Gym.git $GYM_DIR
git -C $GYM_DIR checkout 1ea6b75496c97bf52cdc4578bf34afc4fc5e9e7a
uv pip install --python $PYTHON \
    -r $MILES_DIR/examples/experimental/nemo-gym-workspace-assistant/workplace-requirements.txt
set -gx PYTHONPATH "$GYM_DIR:$MILES_DIR/examples/experimental/nemo-gym-workspace-assistant:$MILES_DIR:$PYTHONPATH"
```

The Gym checkout is required on the **CPU resource-service host** and wherever
you generate/validate data. Its lightweight Workplace modules are imported
directly; you do not need to start the full Gym server stack. Install the Miles
example on both Ray nodes.

Download the BF16 model to the same location on both nodes (or a shared disk):

```fish
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
    --local-dir $MODEL_DIR/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```

The launcher loads HF weights through Megatron Bridge; no offline checkpoint
conversion is needed. MTP training and speculative decoding are disabled.

## 2. Generate data, then convert it

Use the public [data synthesis and calibration scripts](https://gist.github.com/Shi-Dong/37315073940a57e3c3a0bddbf5652b77).
They generate executable objectives from Gym's stock office, validate reference
actions, rewrite the instructions using a user-selected OpenAI model, and
optionally calibrate them against an OpenAI-compatible policy endpoint.
Follow that Gist's README for installation and commands. API keys remain in a
protected file outside the dataset. The generator's model name is configurable.

For the harder set, use `hard_tasks.py --count 2000`, then synthesize and validate.
Calibration with `--samples 8 --max_steps 24` records eight attempts per task.
The reference experiment selected 1,661 tasks with both successes and failures.
Selection is optional: all-0/all-1 groups have no relative GRPO reward signal,
and training-set calibration is not a held-out evaluation.

Place your selected **native Gym JSONL** at `$WORKPLACE_DATA/native_train.jsonl`.
Rows contain `id`, `category`, `responses_create_params` (input and tools),
and `ground_truth`. Keep this native file for the resource service. Convert a
separate policy-only file for Miles:

```fish
$PYTHON $MILES_DIR/examples/experimental/nemo-gym-workspace-assistant/prepare_workplace.py \
    --source $WORKPLACE_DATA/native_train.jsonl \
    --target $WORKPLACE_DATA/workplace_train.jsonl \
    --max_turns 24
```

The converter preserves task IDs, instructions, and tool schemas, and removes
reference actions and synthesis provenance from the training payload.
The launcher controls temperature and token budgets. Use the same native task
catalog for the service as you used to produce the policy file.

## 3. Start the CPU resource service

In a persistent terminal on the trainer or another trusted host:

```fish
$PYTHON $MILES_DIR/examples/experimental/nemo-gym-workspace-assistant/workplace_server.py \
    --dataset $WORKPLACE_DATA/native_train.jsonl \
    --host 0.0.0.0 --port 8211
```

Keep this service on a trusted network: it has no authentication. Its URL must
be reachable from the Ray rollout-manager process. Check `GET /health` for
the task count and active/graded episode counts. The server uses one process;
do not add web workers because episode state is in memory.

## 4. Join the two Ray nodes and launch

On the trainer, set `MASTER_ADDR` to its address reachable from the other node:

```fish
set -gx MASTER_ADDR "REPLACE_WITH_TRAINER_IP"
ray start --head --node-ip-address $MASTER_ADDR --port 6379 \
    --num-gpus 8 --disable-usage-stats
```

On the rollout node, join that cluster:

```fish
ray start --address "REPLACE_WITH_TRAINER_IP:6379" --num-gpus 8 --disable-usage-stats
```

Save the following JSON as `/data/workplace/launcher.json`, replacing paths
and the service address. These must match the files you prepared above.

```json
{
  "model_dir": "/data/models",
  "data_dir": "/data/workplace",
  "data_file": "workplace_train.jsonl",
  "output_dir": "/data/workplace-run",
  "hf_cache_dir": "/data/hf-cache",
  "megatron_path": "/path/to/Megatron-LM",
  "verifier_url": "http://<resource-host>:8211"
}
```

From the trainer, with the same `MASTER_ADDR`:

```fish
set -gx MILES_SCRIPT_EXTERNAL_RAY 1
$PYTHON $MILES_DIR/examples/experimental/nemo-gym-workspace-assistant/run_nemotron35_workplace.py \
    --config /data/workplace/launcher.json
```

The launcher uses Miles' standard W&B configuration when `WANDB_API_KEY` is
set in the environment; do not put credentials in the JSON or commit them.
Metrics are also available in the Miles dashboard. Checkpoints and full rollout
traces go under `output_dir`. The launcher uses the usual Miles cleanup preamble;
run it on dedicated training nodes, not alongside unrelated inference services.

Default recipe (configurable training fields are listed in `ScriptArgs`; the
turn limit is set during conversion, and this recipe requires `retract`):

| Setting | Value |
| --- | --- |
| Learning rate | 3e-7 |
| Prompts × samples per prompt | 8 × 16 |
| Global batch | 128 trajectories |
| Updates / checkpoint interval | 1,000 / 100 |
| Response budget / total context | 65,536 / 81,920 tokens |
| Agent turn limit | 24, set during data conversion |
| Session server / routing replay | v1 / enabled |
| Pause generation mode | retract |
| KL-loss coefficient / entropy coefficient | 0 / 0 |

For a short first run, set `"num_rollout": 3, "save_interval": 3` in a new
output directory. The launcher rejects abort mode: the original run encountered
incomplete routing data with that combination. Current Miles also warns about
routing-replay payload size and SGLang retract-mode limitations. This example
does not claim to fix them; retain the batch-overlapped driver's weight-update
boundaries and inspect traces after the first synchronization.

## 5. Check the first few updates

- Confirm nonzero gradients and changing weights, not just increasing step IDs.
- Inspect several saved trajectories: model actions and tool feedback should
  alternate, and resource-service errors should not become zero-reward samples.
- Check reward, truncation, response length, and diversity within each group.
- `--use-rollout-logprobs` is enabled. Train/rollout log-probability difference
  and KL diagnostics can therefore be zero by construction; those zeros alone
  do not demonstrate numerical agreement between engines.

The native reward is final-state matching, not a judgment of response prose.
There is no extra format or truncation penalty in this Workplace recipe.
The adapter marks transport/verification failures as aborted; the dynamic filter
drops their entire prompt group. If the service stays unhealthy, fix it rather
than treating repeated discarded groups as poor model performance.
