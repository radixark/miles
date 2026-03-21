# Agent V2: Generalized Agent-Environment RL Training with Harbor

A unified pipeline for training agents on **mixed datasets** — SWE-bench, Terminal-Bench, custom tasks, etc. — through a single endpoint. Uses **TITO (Token In Token Out)** through SGLang's `/v1/chat/completions` for exact token-level training signals.

**Agent orchestration and grading** are handled by [Harbor](https://github.com/harbor-framework/harbor). Harbor provides unified rollout + grading in a single `Trial.run()` call. The server is **task-type agnostic** — all differentiation (environment, grading harness) is encoded in each task's 4 files (instruction.md, Dockerfile, test.sh, task.toml).

## Architecture

```
Docker Network (swe-net)
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│ Miles container (GPU)              │       │ Harbor container (agent-env)       │
│                                    │       │                                    │
│  Ray job → train.py                │       │  server.py (port 11000)            │
│    ├─ MegatronTrainRayActor (×N)  │       │    Wraps Harbor Trial API          │
│    ├─ SGLangEngine (×N, 1/GPU)    │       │    Task-type agnostic              │
│    └─ RolloutManager               │       │                                    │
│                                    │       │                                    │
│  agentic_tool_call.generate        │       │                                    │
│  (miles/rollout/generate_hub/)     │       │                                    │
│    1. Create session on Router     │       │                                    │
│    2. Call agent_function.run ─────────────►│  3. Harbor Trial.run():            │
│    5. Collect records              │       │     a. Start Docker (task image)   │
│    6. Build training samples       │       │     b. Install agent               │
│    7. Merge agent metadata         │       │     c. Run agent loop ────────┐   │
│                                    │       │                               │   │
│  Miles Router (port 30000)         │       │                               │   │
│    /sessions/{id}/v1/chat/         │       │                               │   │
│      completions                   │       │                               │   │
│    - Proxies to SGLang engines     │◄──────────── LLM via OPENAI_API_BASE │   │
│    - Records request + response    │       │     d. Run verifier (test.sh)     │
│      with TITO data                │       │     e. Return TrialResult         │
│                                    │       │                                    │
│  SGLang engines (1 per GPU)        │       │  4. Returns:                       │
│    /v1/chat/completions            │       │     reward, exit_status,           │
│    - Applies chat template         │       │     agent_metrics, eval_report     │
│    - Returns input_token_ids       │       │                                    │
│    - Returns token_id per logprob  │       │  Harbor spawns Docker containers   │
│                                    │       │  from task images (on swe-net)     │
│  Megatron (training, GRPO)         │       │                                    │
└───────────────────────────────────┘       └───────────────────────────────────┘
```

## Files

| File | Description |
| --- | --- |
| `run.sh` | Training launcher — handles Ray lifecycle, model loading, and job submission |
| `server.py` | FastAPI server wrapping Harbor Trial API — deploy in the agent-env container |
| `swe_agent_function.py` | Custom agent function — dispatches to Harbor server, returns env metadata |
| `generate.py` | Reward function, agent metrics aggregation, `RolloutFn` |
| `download_and_process_data.py` | Download from HuggingFace or local JSONL, convert to Miles format |
| `prepare_harbor_tasks.py` | Convert Miles JSONL to Harbor task directories (generic fallback) |

## Step-by-Step Setup

### Prerequisites

- Two machines: one with at least 8 x H200 GPUs for launching the training job, and another CPU-only machine for hosting the agent server.  The GPU machine must have Docker with GPU support (nvidia-container-toolkit).
- GLM-4.7-Flash Model weights downloaded on the GPU machine (e.g. `zai-org/GLM-4.7-Flash`).
- Docker CLI installed on the CPU machine via `docker-ce-li`.  [Installation guide](https://docs.docker.com/engine/install/ubuntu/)
- Python virtual environment management tool `uv`.  This can be installed via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Recommended transformer version: `pip install git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa`

### Step 1: start agent server (on the CPU machine)

The following commands assume that the initial working directory is `$CWD`.  The agent server will be hosted on port 11000 of the CPU machine.

```bash
# Clone the miles repo
git clone git@github.com:radixark/miles.git

# Clone the forked harbor repo
git clone git@github.com:radixark/harbor-private.git

# Create a venv named "agent-server" and activate it
cd $CWD/harbor-private && uv venv agent-server
source agent-server/bin/activate  # For fish shell: source agent-server/bin/activate.fish

# Install harbor
uv pip install harbor

# Download and register the SWE-Gym tasks to $HARBOR_TASKS_DIR
cd $CWD/harbor-private/adapters/swe-gym && uv sync
uv run run_adapter.py --task-dir $HARBOR_TASKS_DIR

# Start the agent server
export OPENAI_API_KEY="dummy"
uv run python $CWD/miles/examples/experimental/swe-agent-v2/server.py \
    --port 11000 \
    --max-concurrent 8 \
    > /tmp/server.log 2>&1
```

You can verify that the agent server is running by `curl -s http://localhost:11000/health`.  If you see `{"status":"ok"}`, the agent server is ready to go.

### Step 2: Start the Miles container (on the GPU machine), download data, and prepare model weights.

The following command starts a docker container named `miles` off image `radixark/miles:latest`.
Note that the command masks the `miles` repo on the image with the `miles` repo on the host machine.
This isn't always necessary.  Do it at your own discretion.
```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  -v $LOCAL_MILES_DIR:/root/miles \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --name miles \
  radixark/miles:latest \
  /bin/bash
```

Inside the docker container `miles`,
```bash
# Install the appropriate `transformers` package
pip install git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa

# Download SWE-Gym data and convert to Miles format
python /root/miles/examples/experimental/swe-agent-v2/download_and_process_data.py --input SWE-Gym/SWE-Gym --output /root/swe.jsonl

# If needed, download GLM-4.7-Flash model weights
huggingface-cli download zai-org/GLM-4.7-Flash --local-dir /root/GLM-4.7-Flash

# Converting HuggingFace model weights to Megatron format for training
export PYTHONPATH="/root/Megatron-LM"
source /root/miles/scripts/models/glm4.7-flash.sh
python /root/miles/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-4.7-Flash \
    --save /root/GLM-4.7-Flash_torch_dist
```

Each Harbor task directory contains 4 files:
- `instruction.md` — agent prompt
- `task.toml` — config (timeout, resources)
- `environment/Dockerfile` (or `docker-compose.yaml`) — container image
- `tests/test.sh` — grading logic

### Step 3: Launch training

Inside the `miles` docker container in the GPU machine,

```bash
# Inside miles container:
cd /root/miles/examples/experimental/swe-agent-v2

# Debug mode (small batch, quick pipeline verification)
bash run.sh --mode debug --num-gpus 8 --ep 8 \
  --prompt-data /root/mixed.jsonl \
  --sglang-tool-call-parser glm47 \
  --sglang-reasoning-parser glm45

# Full training
bash run.sh --num-gpus 8 --ep 8 \
  --prompt-data /root/mixed.jsonl \
  --sglang-tool-call-parser glm47 \
  --sglang-reasoning-parser glm45
```

What `run.sh` does:
1. Kills stale sglang/ray processes
2. Starts a Ray head node with `--num-gpus` GPUs
3. Sources the model architecture script (e.g. `glm4.7-flash.sh`)
4. Submits a Ray job running `train.py` with all configured arguments


### Step 8 (Optional): Start the trace-viewer

[radixark/trace-viewer](https://github.com/radixark/trace-viewer) visualizes agent trajectories. Run it inside agent_env where Harbor saves trial artifacts:

```bash
docker exec -d agent_env bash -c \
  'python3 /root/trace-viewer/server.py --dir /root/trials \
   --host 0.0.0.0 --port 8081 > /tmp/trace_viewer.log 2>&1'
```

If the container isn't directly reachable, set up a port forwarder on the host:

```bash
# On the Docker host — forward host:8081 to agent_env:8081
# (replace 172.18.0.4 with agent_env's IP from: docker inspect agent_env -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
socat TCP-LISTEN:8081,fork,reuseaddr TCP:172.18.0.4:8081
```

Then open `http://<host>:8081` in a browser.

## Key Configuration

### run.sh defaults vs overrides

| Parameter | Default | Debug mode override |
| --- | --- | --- |
| `--num-rollout` | 3000 | 50 |
| `--rollout-batch-size` | 8 | 4 |
| `--n-samples-per-prompt` | 8 | 4 |
| `--rollout-max-response-len` | 8192 | 4096 |
| `--global-batch-size` | 64 | 16 |
| `--max-tokens-per-gpu` | 2048 | 1024 |

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_SERVER_URL` | `http://agent_env:11000` | Harbor server URL |
| `AGENT_MODEL_NAME` | `model` | Model name passed to agents |
| `AGENT_MAX_CONCURRENT` | `8` | Max concurrent Harbor trials |
| `HARBOR_TASKS_DIR` | `/root/harbor_tasks` | Root directory containing task subdirectories |
| `MILES_ROUTER_EXTERNAL_HOST` | `$(hostname)` | Hostname for agent containers to reach Miles Router |
| `MILES_HOST_IP` | `$(hostname)` | IP/hostname for inter-container communication |

### Model-specific arguments

These are passed as CLI args to `run.sh` (not defaults, since they vary per model):

| Model | Tool call parser | Reasoning parser |
| --- | --- | --- |
| GLM-4.7-Flash | `--sglang-tool-call-parser glm47` | `--sglang-reasoning-parser glm45` |
| Qwen3 | `--sglang-tool-call-parser qwen25` | (none) |

## How It Works

1. **Session creation**: `agentic_tool_call.generate` creates a session on Miles Router
2. **Dispatch to agent**: calls `swe_agent_function.run()` which POSTs to the Harbor server
3. **Harbor Trial**: `server.py` creates a `TrialConfig` and runs `Trial.run()`:
   - Starts a Docker container from the task's Dockerfile
   - Installs and runs the agent (determined by `agent_name` in metadata)
   - Agent calls back to Miles Router at `OPENAI_API_BASE` for model inference
   - Runs the verifier (`test.sh`) and returns `TrialResult` with reward
4. **TITO recording**: Miles Router proxies each `/v1/chat/completions` to SGLang and records exact token IDs and logprobs
5. **Sample building**: Records are converted to training `Sample`s with token IDs, logprobs, loss masks
6. **Training**: GRPO policy update using Megatron, then weights synced back to SGLang engines

### Multi-turn merge (TITO mode)

When TITO is working, `merge_samples()` concatenates multi-turn conversations:

```
Turn 1: [prompt_tokens] [response_tokens_1]                                    -> Sample 0
Turn 2: [prompt_tokens] [response_tokens_1] [observation_tokens] [resp_tokens_2] -> Sample 1

merge_samples() ->
  tokens:    [prompt] [resp1] [obs] [resp2]
  loss_mask: -------- [1]*r1  [0]*o [1]*r2
  logprobs:  -------- [real]  [0.0] [real]
```

Without TITO, use `--generate-multi-samples` to skip merge and train on per-turn samples instead (current default in `run.sh`).

## Troubleshooting

### Harbor containers can't reach Miles Router

Agent containers need to resolve the Miles container's hostname. Ensure:
- All containers are on the same Docker network (`swe-net`)
- `MILES_ROUTER_EXTERNAL_HOST` is set to the Miles container name (e.g. `miles-maocheng`)
- Task directories were created with `--docker-network swe-net`

### `TaskNotFound` error

The task directory for the given `instance_id` doesn't exist under `HARBOR_TASKS_DIR`. Run the appropriate Harbor adapter or `prepare_harbor_tasks.py` first.

### SGLang engines OOM (`Not enough memory`)

- Ensure all GPUs are clean before starting (`nvidia-smi` shows 0 MiB)
- Kill stale processes: `pkill -9 sglang; ray stop --force; pkill -9 ray`
- `run.sh` does this automatically, but leftover processes from crashed jobs may persist

### `b.tokens must start with a.tokens` assertion error

Multi-turn merge fails due to BPE re-tokenization inconsistency. Use `--generate-multi-samples` (already default in `run.sh`) to skip merge and train on per-turn samples.

### Trace-viewer shows no trajectories

- Trial artifacts are saved inside the **agent_env** container (not miles), at the path Harbor uses (default: `./trials/` relative to where `server.py` runs)
- Point the trace-viewer at the correct directory inside agent_env
- Restart the trace-viewer after clearing old data (it caches in memory)
