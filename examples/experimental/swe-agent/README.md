# SWE-agent Training Example

## Introduction

This example demonstrates SWE-agent RL training in Miles using:
- `mini-swe-agent` for agent execution
- SWE-Gym data for training prompts
- SWE-bench style evaluation via `swegym`

The implementation depends on the `mini-swe-agent` submodule in this directory.
Submodule reference:
- mini-swe-agent: https://github.com/yueming-yuan/nv-mini-swe-agent/tree/miles-swe-agent

## Prepare environment

### 1. Update submodules
```bash
git submodule update --init --recursive
```

### 2. Docker requirements
SWE tasks run inside per-instance Docker containers. The Miles container must be able to launch Docker containers.

Required when launching the Miles container:
```bash
-v /var/run/docker.sock:/var/run/docker.sock
```

Install Docker CLI in the Miles container if needed:
```bash
apt update && apt install -y docker.io
```

### 3. Install Miles and SWE eval dependency
From the Miles repo root:
```bash
pip install -e . --no-deps
pip install "swegym @ git+https://github.com/sdevare-nv/nv-SWE-Bench-Package.git@31e1cb8f0241da1707d00faa633c3d6ce1a8ba3b"
```

## Prepare model and data

### 1. Download model checkpoint
```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/qwen3-4B-Instruct-2507
```

### 2. Convert HF checkpoint to Megatron torch_dist format
```bash
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/qwen3-4B-Instruct-2507 \
    --save /root/qwen3-4B-Instruct-2507_torch_dist
```

### 3. Download and process SWE-Gym prompts
```bash
cd examples/experimental/swe-agent
python3 download_and_process_data.py --input SWE-Gym/SWE-Gym --output /root/swe_train.jsonl
```

## Run training

From the Miles repo root:
```bash
bash examples/experimental/swe-agent/run-qwen3-4b-instruct.sh
```

Optional Weights & Biases:
```bash
export WANDB_KEY=<your_wandb_api_key>
```

## How this example works

1. `generate_with_swe_agent.generate` creates a router session.
2. `mini-swe-agent` runs task steps in a Docker environment.
3. Router session records are converted to training samples (`tokens`, `logprobs`, `loss_mask`).
4. SWE evaluation result is converted to reward and attached to sample metadata.

## Troubleshooting

1. First startup can be slow because task-specific Docker images are pulled on demand.
2. Evaluation can take time (default timeout is 10 minutes); while eval is running, wait for completion logs.
3. If runs stop with CUDA OOM, reduce sequence pressure first (for example, `--max-tokens-per-gpu`, `--rollout-max-response-len`, or rollout batch sizing).

## Metrics
```
agent/turns_mean, agent/turns_sum - Turn counts
agent/tool_calls_mean, agent/tool_calls_sum - Tool call counts
agent/total_time_mean/max/min - Total time statistics
agent/model_query_time_sum_mean - Avg total model time per rollout
agent/env_execution_time_sum_mean - Avg total env time per rollout
agent/eval_time_mean - Avg evaluation time
agent/overhead_time_mean - Avg overhead time
agent/time_per_turn - Avg time per turn
agent/model_query_time_avg - Avg model query time per turn
agent/env_execution_time_avg - Avg env execution time per turn
agent/model_time_ratio, agent/env_time_ratio, agent/eval_time_ratio - Time ratios
```
