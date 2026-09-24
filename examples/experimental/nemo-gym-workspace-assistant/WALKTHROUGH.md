# Train a workplace assistant with Miles

Teach **Nemotron 3.5 Lightning** to complete office tasks in **NeMo Gym**: search email, update project records, change calendar events, and check the results. Miles runs the training; Gym grades the work.

New to Miles? Begin with the [Miles Quick Start](https://miles.radixark.com/docs/getting-started/quick-start). Once you have seen that reward curve climb, come back here and put the same loop to work on a real agent task.

**The workflow:** create tasks → try the untrained model → choose tasks → train → inspect results.

## Before you start

- **GPUs:** 2 nodes × 8 GPUs (check the [Miles documentation](https://miles.radixark.com/docs/getting-started/installation#hardware-requirements) for supported hardware).
- **Storage:** reserve **8–10 TB of free SSD/NVMe space on the trainer** and **1 TB on the rollout node**, keeping all checkpoints and full training traces.
- **Time:** plan for **3–5 hours from data generation to launching training**, or **2–3 days through all 1,000 training updates**.
- An LLM API key, to write the task instructions.
- A [Weights & Biases](https://wandb.ai) account, to track training on the W&B dashboard.
- `jq`, for the task-selection command.

The expandable sections contain the setup commands. Paths are examples; use your own mounted storage. Commands use Fish shell.

<details>
<summary>One-time setup: helper scripts and paths</summary>

This example needs three repositories beyond the Quick Start: the [task-generation helpers](https://github.com/radixark/miles/tree/main/examples/experimental/nemo-gym-workspace-assistant) and [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym/tree/1ea6b75496c97bf52cdc4578bf34afc4fc5e9e7a). Keep the SGLang and Megatron-LM installation from your training environment.

Place the helper folder at `/workspace/workplace-data-tools`, Gym at `/workspace/Gym`, and the tested Miles checkout at `/workspace/miles`.

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

Install the data helpers in their own environment, separate from the training stack:

```fish
cd $DATA_CODE
uv venv --python 3.12
uv pip install -e '.[test]'
uv pip install --python $DATA_PY -r $EXAMPLE/workplace-requirements.txt
uv pip install --python $TRAIN_PY -r $EXAMPLE/workplace-requirements.txt
set -gx PYTHONPATH $GYM_ROOT
$DATA_PY -m pytest -q test_contract.py test_hard_tasks.py
```

Download the model to the same path on **both** GPU nodes:

```fish
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
    --local-dir $MODEL_ROOT/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```

No checkpoint conversion here: the launcher loads Hugging Face weights directly through Megatron Bridge.

</details>

## 1. Generate tasks

Each task has an initial office state, a plain-English request, and a reference sequence of actions used only for grading. The generator builds the actions first, an LLM writes the request, and validation rejects incomplete solutions. The model being trained never sees the reference answers.

<details>
<summary>Generate and validate 2,000 tasks</summary>

Run in the data-tools terminal. Store your API key in a protected file and pass it with `--key_file`; choose the writer model with `--model`.

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

**Result:** `$DATA_ROOT/dataset/train.jsonl` with 2,000 candidate tasks. 30–60 minutes.

</details>

## 2. Calibrate difficulty

Let the untrained model try each task **eight times** and keep the tasks it solved 1–7 times. GRPO learns by comparing attempts at the same task, so tasks that always or never succeed teach nothing. We kept **1,661 of 2,000**; your count will differ.

<details>
<summary>Start a model endpoint and run calibration</summary>

Serve the model on one GPU, in a terminal with the training environment:

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

This is 16,000 attempts. The command above runs 8 at a time on one GPU; we used 16 model replicas and 512 concurrent attempts to finish in 2–3 hours. Keep `--samples 8` whatever you scale.

Select the mixed-success tasks with `jq`:

```fish
jq -c --slurpfile scores "$DATA_ROOT/calibration/task_scores.jsonl" \
    'INDEX($scores[] | select(.completed_rollouts == 8 and .successes > 0 and .successes < 8); .id) as $keep
     | select($keep[.id | tostring] != null)' \
    "$DATA_ROOT/dataset/train.jsonl" > "$DATA_ROOT/train.jsonl"
```

Keep the original tasks and all calibration results too.

</details>

## 3. Connect NeMo Gym

Every attempt starts in a fresh office. Tool calls run Gym's code and return real results; at the end Gym returns **reward 1** if the required office state is reached, **0** otherwise.

The rollout node must be able to reach port 8211 on the trainer. Keep the simulator on a trusted network: it has no authentication and holds grading data.

<details>
<summary>Prepare the training file and start the simulator</summary>

Convert the selected tasks into Miles' input format:

```fish
$TRAIN_PY $EXAMPLE/prepare_workplace.py \
    --source $DATA_ROOT/train.jsonl --target $RUN_DIR/workplace_train.jsonl
```

On the trainer node, start the simulator in its own terminal and leave it running:

```fish
set -gx PYTHONPATH $GYM_ROOT
$DATA_PY $EXAMPLE/workplace_server.py \
    --dataset $DATA_ROOT/train.jsonl --port 8211
```

</details>

## 4. Launch training

Stop the calibration endpoint to free its GPUs. Use the setup below to connect the two nodes with Ray and save the training configuration.

<details>
<summary>Two-node setup and training configuration</summary>

Connect the two nodes with Ray, each command in its own terminal using the training environment:

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

Save as `$RUN_DIR/launcher.json`. Replace the example paths and `TRAINER_IP`:

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

In the trainer's launch terminal, set the environment before launching:

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

The standard Miles W&B helper enables logging when `WANDB_API_KEY` is set and uses project `miles-run_nemotron35_workplace`. Keep `pause_generation_mode` set to `retract`: `abort` crashed in the versions used for this experiment.

</details>

Then launch from the **trainer node**:

```fish
$TRAIN_PY $EXAMPLE/run_nemotron35_workplace.py \
    --config $RUN_DIR/launcher.json > $RUN_DIR/train.log 2>&1
```

That's it. A few things the launcher already does for you:

- Each batch is 8 tasks × 16 attempts = **128 attempts**; 1,000 updates in total.
- Checkpoints land in `output_dir` every 100 updates, about 442 GB each.

## 5. Check that training is useful

`rollout/raw_reward` on the W&B dashboard is the number to watch: the fraction of tasks completed per batch, climbing as the policy improves.

Calibration and training here use the same task pool. To measure performance on new tasks, reserve a separate evaluation set.

## What's happening

A Miles job combines two engines: [SGLang](https://github.com/sgl-project/sglang) generates attempts from the current policy (the *rollout*), and Megatron-LM updates the policy from those attempts (the *training*). In this recipe they run on separate nodes, and Miles ships the updated weights across after every step.

1. Sample 8 tasks and let SGLang generate 16 attempts per task. Each attempt is a conversation: the model calls a tool, Gym runs it and returns the result, the model decides what to do next, for up to 24 turns.
2. Score every attempt. Gym checks the final office state and returns 1 or 0.
3. Compute the GRPO objective from the scores and step the optimizer. GRPO compares the 16 attempts within each task, so a success where siblings failed earns a large positive advantage.
4. Sync the updated weights to the rollout node, and go again.

Learn more about [Miles](https://miles.radixark.com/), explore the [documentation](https://miles.radixark.com/docs), read the [technical blog](https://www.lmsys.org/blog/2026-08-18-miles-v0-1), and contribute to [GitHub](https://github.com/radixark/miles).
