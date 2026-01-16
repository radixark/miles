# SWE-agent training Example

## Introduction

This is an example for SWE-agent training. This example uses NVIDIA's `mini-swe-agent` as the agent implementation, SWE-Gym as the training data, and SWE-bench as the evaluation harness.

The implementation of this example is partially in the submodule below:
- mini-swe-agent: https://github.com/yueming-yuan/nv-mini-swe-agent/tree/miles-swe-agent

## Prepare environment

### Update submodules
```bash
git submodule update --init --recursive
```

### Docker settings
To run SWE-agent, the Miles container must be able to launch temporary Docker containers for each task environment. This requires mounting the Docker socket.

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /mnt/data/cache/huggingface:/root/.cache/huggingface \
  -v /mnt/data:/data \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --network host \
  --name miles_<your_name> \
  radixark/miles:latest \
  /bin/zsh
```
note: `-v /var/run/docker.sock:/var/run/docker.sock` is required for Docker-in-Docker SWE environment execution.

### Installation

Inside the **Miles docker**, install the Docker CLI and the SWE-Gym harness:

```bash
# Install Docker CLI
apt update && apt install -y docker.io

# Install SWE-Gym harness
pip install "swegym @ git+https://github.com/sdevare-nv/nv-SWE-Bench-Package.git@31e1cb8f0241da1707d00faa633c3d6ce1a8ba3b"
```

## Preparing data
Download **SWE-Gym** data from Hugging Face and convert it to Miles' prompt data format:
```bash
cd examples/experimental/swe-agent
python3 download_and_process_data.py --input SWE-Gym/SWE-Gym --output /root/swe_train.jsonl
```

## Running train
Launch the training directly from the Miles root directory:
```bash
bash examples/experimental/swe-agent/run-qwen3-4b-instruct.sh
```

## Troubleshooting
1. **Slow Startup:** The first time a SWE environment is created, it may be slow because each SWE-Gym task requires a specific Docker image, and `docker pull` takes time.
2. **Evaluation Timeout:** Sometimes the environment may be slow during evaluation. The default timeout is 10 minutes. If logs show `[EVAL]<instance> Running eval`, please wait for completion.
3. **Ray Stability:** The rollout process uses background threads to ensure the Ray cluster remains responsive during long-running agent tasks.

## Metrics
```
agent/turns_mean, agent/turns_sum - Turn counts
agent/tool_calls_mean, agent/tool_calls_sum - Tool call counts
agent/total_time_mean/max/min - Total time statistics
agent/model_query_time_sum_mean - Avg total model time per rollout
agent/env_execution_time_sum_mean - Avg total env time per rollout
agent/eval_time_mean - Avg evaluation time
agent/time_per_turn - Avg time per turn
agent/model_query_time_avg - Avg model query time per turn
agent/env_execution_time_avg - Avg env execution time per turn
agent/model_time_ratio, agent/env_time_ratio - Time ratios
```
