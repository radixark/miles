# Train a workplace assistant with Miles

Adapted from the [original walkthrough](https://gist.github.com/Shi-Dong/c8746fee78bc3a87cbc531ade514c13c).
Commands below target this checkout's example. The recorded run and time/storage
estimates refer to the historical experiment; GPU validation on current backend
versions remains necessary. See [the recipe reference](README.md) for
compatibility notes.

Teach **Nemotron 3.5 Lightning** to complete office tasks: search email, update project records, change calendar events, and check the results.

The model works in a simulated office provided by **NeMo Gym**. It calls a tool, reads the response, and decides what to do next. Gym checks whether the requested changes were actually made; Miles uses that score to train the model.

[Recorded training run → W&B](https://wandb.ai/radixarkai/nemotron35-workplace-assistant/runs/kk6omvbp)

**The workflow:** create tasks → try the untrained model → choose tasks → train → inspect results.

## Before you start

Start with the [Miles Quick Start](https://miles.radixark.com/docs/getting-started/quick-start) for the container setup and your first RL run. This guide adds a custom Workplace Assistant example to that foundation.

For this experiment, we used **two nodes with eight H200 GPUs each**: one generates attempts, the other trains. You also need an OpenAI API key for writing tasks, a W&B account for tracking training, and jq for the selection command.

- **Storage:** reserve **8–10 TB of free SSD/NVMe space on the trainer** and **1 TB on the rollout node**, keeping all checkpoints and full training traces.
- **Time:** plan for **3–5 hours from data generation to launching training**, or **2–3 days through all 1,000 training updates**.

These time estimates assume the software and model downloads are ready, and calibration uses the original **16-GPU serving setup**. The single-GPU calibration example below will take longer. Provisioning, troubleshooting, and uploading backups are additional.

<details>
<summary>Where the storage and time estimates come from</summary>

**Storage:** ten checkpoints at about 442 GB each use **4.4 TB**. Full training traces are projected to use another **2 TB**: they occupied about 626 GB after 308 updates. The 8–10 TB trainer budget adds room for model weights, caches, logs, and temporary files. The rollout node does not store training checkpoints. Sizes use decimal GB/TB; trace growth depends on trajectory length.

**Time budget on the original hardware:**

| Stage | Planning estimate |
|---|---|
| Generate and validate 2,000 tasks | 30–60 minutes |
| Calibrate all tasks, eight attempts each | 2–3 hours with 16 model replicas |
| Train for 1,000 updates | 50–55 hours |

Training completed 423 updates in about 21.6 hours, which projects to about 51 hours for 1,000 updates. The experiment was stopped before 1,000, so the full-run duration and final trace size are **projections, not measured completion results**. API throughput, storage speed, and rollout lengths affect these estimates.

</details>

The expandable sections contain the setup commands. Paths are examples; use your own mounted storage. Commands use Fish shell.

<details>
<summary>One-time setup: helper scripts and paths</summary>

Use the [task-generation helpers](https://gist.github.com/Shi-Dong/37315073940a57e3c3a0bddbf5652b77), the [Miles Workplace example in this checkout](README.md), and [this Gym revision](https://github.com/NVIDIA-NeMo/Gym/tree/1ea6b75496c97bf52cdc4578bf34afc4fc5e9e7a). Download all files in the helper Gist into one directory. These helper scripts are required; they are not all part of the basic Quick Start. Install Gym at the linked revision.

Place the helper folder at `/workspace/workplace-data-tools`, Gym at `/workspace/Gym`, and your Miles checkout at `/workspace/miles`. Keep the compatible SGLang and Megatron installation from your training environment. The original run's source versions are linked at the end.

Set these paths in each relevant terminal:

```fish
set DATA_CODE /workspace/workplace-data-tools
set GYM_ROOT /workspace/Gym
set MILES_ROOT /workspace/miles
set MEGATRON_ROOT /workspace/Megatron-LM
set EXAMPLE $MILES_ROOT/examples/experimental/nemo-gym-workspace-assistant
set DATA_ROOT /data/workplace
set RUN_DIR /outputs/workplace-run
set MODEL_ROOT /models
set DATA_PY $DATA_CODE/.venv/bin/python
# Python from your existing Miles training environment:
set TRAIN_PY /venvs/training/bin/python
mkdir -p $DATA_ROOT $RUN_DIR
```

Install the data helpers separately from the training environment:

```fish
cd $DATA_CODE
uv venv --python 3.12
uv pip install -e '.[test]'
uv pip install --python $DATA_PY -r $EXAMPLE/workplace-requirements.txt
uv pip install --python $TRAIN_PY -r $EXAMPLE/workplace-requirements.txt
set -gx PYTHONPATH $GYM_ROOT
$DATA_PY -m pytest -q test_contract.py test_hard_tasks.py
```

Download the original model to the same path on both GPU nodes:

```fish
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
    --local-dir $MODEL_ROOT/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```

This launcher loads the Hugging Face weights through Megatron Bridge, so it does not need the Quick Start's separate checkpoint-conversion step.

</details>

## 1. Create tasks with checkable outcomes

Start with tasks such as updating project assignments and notifying their owners. Each task needs an initial office state, a clear request, and a known sequence of actions that completes it.

Our generator builds those reference actions first. **GPT 5.6 Luna writes the user-facing instructions** while preserving the rules. Validation checks the reference actions and catches incomplete solutions. The model being trained never sees the reference answers.

<details>
<summary>Generate and validate 2,000 tasks</summary>

Run in the data-tools terminal. Store your API key in a protected file and point the command at it.

```fish
set -gx PYTHONPATH $GYM_ROOT
$DATA_PY $DATA_CODE/hard_tasks.py \
    --output_dir $DATA_ROOT/specs --count 2000 --workers 12

$DATA_PY $DATA_CODE/synthesize.py \
    --specs $DATA_ROOT/specs/specs.jsonl \
    --output_dir $DATA_ROOT/dataset \
    --key_file /secrets/openai-api-key --model gpt-5.6-luna \
    --workers 12 --batch_size 8

$DATA_PY $DATA_CODE/validate_export.py --dataset $DATA_ROOT/dataset
```

**Result:** `/data/workplace/dataset/train.jsonl`, containing the candidate tasks.

</details>

## 2. Check the difficulty before training

Let the original Nemotron model try each task **eight times**. This is *calibration*: it tells us which tasks the model can sometimes solve.

We kept tasks with **1–7 successes out of 8**, giving training both successful and unsuccessful attempts to compare. In this experiment, **1,661 of the 2,000 tasks** met that rule. Your results may differ.

<details>
<summary>Start a model endpoint and run calibration</summary>

On one GPU, in a separate terminal using the training environment:

```fish
set -gx CUDA_VISIBLE_DEVICES 0
$TRAIN_PY -m sglang.launch_server \
    --model-path $MODEL_ROOT/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --served-model-name nemotron35-lightning \
    --host 0.0.0.0 --port 8000 --tp-size 1 --dtype bfloat16 \
    --context-length 81920 --mem-fraction-static 0.7 \
    --max-running-requests 32 --cuda-graph-max-bs 32 \
    --reasoning-parser nemotron_3 --tool-call-parser qwen3_coder \
    --trust-remote-code
```

Once the endpoint is ready, run the calibrator in the data-tools terminal:

```fish
set -gx PYTHONPATH $GYM_ROOT
$DATA_PY $DATA_CODE/calibrate.py \
    --dataset $DATA_ROOT/dataset/train.jsonl \
    --output_dir $DATA_ROOT/calibration \
    --base_url http://MODEL_ENDPOINT_HOST:8000/v1 \
    --tasks 2000 --samples 8 --workers 8 --max_steps 24 \
    --response_budget 65536 --context_length 81920 --temperature 1.0
```

The example starts with eight concurrent attempts. Our experiment used 16 model replicas and 512 concurrent attempts to finish faster; the eight attempts **per task** stayed the same. This stage produces 16,000 attempts and saves the scores and traces.

Select the mixed-success tasks with `jq`:

```fish
jq -c --slurpfile scores "$DATA_ROOT/calibration/task_scores.jsonl" \
    'INDEX($scores[] | select(.completed_rollouts == 8 and .successes > 0 and .successes < 8); .id) as $keep
     | select($keep[.id | tostring] != null)' \
    "$DATA_ROOT/dataset/train.jsonl" > "$DATA_ROOT/train.jsonl"
```

Keep the original tasks and all calibration results too.

</details>

## 3. Connect the office simulator

Each attempt starts with a fresh simulated office. Tool calls execute Gym's code and return the actual results of those operations. At the end, **Gym gives reward 1 if the required office state is reached, or 0 otherwise**.

The small Miles adapter connects this simulator to the model and records the conversation. It trains on the model's responses, leaving tool feedback out of the training loss.

<details>
<summary>Prepare the training file and start the simulator</summary>

Convert the selected tasks into Miles' input format:

```fish
$TRAIN_PY $EXAMPLE/prepare_workplace.py \
    --source $DATA_ROOT/train.jsonl --target $RUN_DIR/workplace_train.jsonl
```

On the trainer node, keep the simulator running in its own terminal. It uses the native task file, which retains the private grading information:

```fish
set -gx PYTHONPATH $GYM_ROOT
$DATA_PY $EXAMPLE/workplace_server.py \
    --dataset $DATA_ROOT/train.jsonl --port 8211
```

The rollout-manager process must be able to reach this service. Keep it on a trusted network: it has no authentication and holds grading data. Infrastructure failures are discarded, rather than graded as model failures.

</details>

## 4. Launch training

Stop the calibration endpoint to free its GPUs. Use the setup below to connect the two nodes and save the training configuration.

Each training batch contains **8 tasks × 16 attempts = 128 trajectories**. GRPO compares the rewards within each task's group. The trainer updates Nemotron, and Miles sends the new weights to the rollout node.

<details>
<summary>Two-node setup and training configuration</summary>

Both nodes need matching code and model paths, plus the prepared Miles training file. Use a fresh output directory for a new run. The example starts from the original model weights.

Ray connects the two machines. In separate terminals using the training environment, replace the address placeholders with addresses the nodes can reach:

```fish
# Trainer node
ray start --head --node-ip-address TRAINER_IP --port 6379 \
    --num-gpus 8 --dashboard-host 0.0.0.0 --disable-usage-stats --block
```

```fish
# Rollout node
ray start --address TRAINER_IP:6379 --node-ip-address ROLLOUT_IP \
    --num-gpus 8 --disable-usage-stats --block
```

Save this as `/outputs/workplace-run/launcher.json`. Replace the example paths and `TRAINER_IP`:

```json
{
  "output_dir": "/outputs/workplace-run",
  "model_dir": "/models",
  "data_dir": "/outputs/workplace-run",
  "megatron_path": "/workspace/Megatron-LM",
  "verifier_url": "http://TRAINER_IP:8211",
  "learning_rate": 3e-7,
  "rollout_batch_size": 8,
  "group_size": 16,
  "global_batch_size": 128,
  "num_rollout": 1000,
  "save_interval": 100,
  "response_length": 65536,
  "context_length": 81920,
  "pause_generation_mode": "retract"
}
```

In the trainer's launch terminal, authenticate with W&B and set the environment before launching:

```fish
set -gx PYTHONPATH $MILES_ROOT:$MEGATRON_ROOT:$EXAMPLE
set -gx MILES_SCRIPT_EXTERNAL_RAY 1
set -gx MILES_NEMOTRONH_KEEP_MTP ''
set -gx CUDA_DEVICE_MAX_CONNECTIONS 1
set -gx MASTER_ADDR TRAINER_IP
cd $MILES_ROOT
# Read your W&B key from a protected file; do not put it in the config or Git:
set -gx WANDB_API_KEY (string trim < /secrets/wandb-api-key)
```

The standard Miles W&B helper enables logging when `WANDB_API_KEY` is set and
uses project `miles-run_nemotron35_workplace`. The old run's W&B project remains
linked for reference; `wandb_team` and `wandb_project` are not fields of this launcher.

The configuration uses learning rate `3e-7`, 1,000 updates, and a checkpoint every 100 updates. Each attempt allows 24 model turns, 65,536 generated tokens in total, and 81,920 context tokens.

The launcher disables multi-token prediction (MTP), enables expert-routing replay, and records traces and dashboard metrics. Keep `pause_generation_mode` set to `retract`: `abort` crashed in the versions used for this experiment.

</details>

Then launch from the **trainer node**:

```fish
$TRAIN_PY $EXAMPLE/run_nemotron35_workplace.py \
    --config $RUN_DIR/launcher.json > $RUN_DIR/train.log 2>&1
```

## 5. Check that training is useful

After startup, inspect the first completed batch:

- **Reward:** `rollout/raw_reward` is the fraction of successfully completed tasks.
- **Learning:** check that gradients are nonzero and rewards vary within some task groups.
- **Behavior:** read a few traces to see sensible tool calls and meaningful feedback. Watch for increasing truncation or environment errors.

Open the [recorded W&B run](https://wandb.ai/radixarkai/nemotron35-workplace-assistant/runs/kk6omvbp) to see the experiment's metrics. The launcher also saves traces and enables the Miles dashboard.

Calibration and training here use the same task pool. To measure performance on new tasks, reserve a separate evaluation set.

<details>
<summary>Reference code and tested versions</summary>

- [Miles Quick Start](https://miles.radixark.com/docs/getting-started/quick-start): the basic setup and training loop.
- [Task-generation helpers](https://gist.github.com/Shi-Dong/37315073940a57e3c3a0bddbf5652b77).
- [Miles Workplace integration](https://github.com/radixark/miles/tree/4e039aed9394f460e92e5703be14832fbacdfea2/examples/experimental/nemo-gym#workplace-assistant).
- [NeMo Gym Workplace environment](https://github.com/NVIDIA-NeMo/Gym/blob/1ea6b75496c97bf52cdc4578bf34afc4fc5e9e7a/resources_servers/workplace_assistant/README.md).
- Tested backend commits: SGLang `ebdf84ca`, Megatron-LM `8c1e0574`, Megatron Bridge `582783a0`.

</details>
