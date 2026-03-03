# SWE-agent Training Example

## Introduction

This example demonstrates SWE-agent RL training in Miles using:
- `mini-swe-agent` for agent execution
- SWE-Gym data for training prompts
- SWE-bench style evaluation via `swegym`

Implementation dependencies:
- Nemo-Gym (Gym): https://github.com/yueming-yuan/Gym/tree/miles-swe-agent
- mini-swe-agent: https://github.com/yueming-yuan/nv-mini-swe-agent/tree/miles-swe-agent

## Prepare environment

### 1. Update submodules
```bash
git submodule update --init --recursive
```

### 2. Docker requirements
SWE tasks run inside per-instance Docker containers. The Miles container must be able to launch Docker containers.

Required when launching Miles:
```bash
-v /var/run/docker.sock:/var/run/docker.sock
```

Install Docker CLI in the Miles container if needed:
```bash
apt update && apt install -y docker.io
```

### 3. Optional two-container setup (environment + Miles)
If you run Gym in a separate container, use a shared Docker network and mount Docker socket into the environment container.

```bash
# create network
docker network create swe-net

# environment container (example)
docker run -itd \
  --name swe_env \
  --shm-size 16g \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/data:/data \
  -v /home/sglang-rl/<your_name>:/workspace \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network swe-net \
  ubuntu:latest \
  /bin/bash

# miles container (example)
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /mnt/data/cache/huggingface:/root/.cache/huggingface \
  -v /mnt/data:/data \
  -v /home/sglang-rl/<your_name>:/workspace \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --network swe-net \
  --name miles_<your_name> \
  radixark/miles:latest \
  /bin/zsh
```

## Installation

### In Miles container
From the Miles repo root:
```bash
pip install -e . --no-deps
pip install "swegym @ git+https://github.com/sdevare-nv/nv-SWE-Bench-Package.git@31e1cb8f0241da1707d00faa633c3d6ce1a8ba3b"
```

### Optional: in environment container (if running Gym server separately)
```bash
cd /
git clone https://github.com/yueming-yuan/Gym
git clone https://github.com/yueming-yuan/nv-mini-swe-agent.git
cd nv-mini-swe-agent && git checkout -b miles-swe-agent origin/miles-swe-agent
cd /Gym && git checkout -b miles-swe-agent origin/miles-swe-agent

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev --group docs

# configure env.yaml
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14
default_host: 0.0.0.0" > env.yaml
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
2. Evaluation can take time (default timeout is 10 minutes); if logs show eval running, wait for completion.
3. If runs stop with CUDA OOM, reduce sequence pressure first (for example, `--max-tokens-per-gpu`, `--rollout-max-response-len`, or rollout batch sizing).

## Metrics

```text
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
