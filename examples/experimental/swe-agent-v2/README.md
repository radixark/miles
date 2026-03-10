# Agent V2: Generalized Agent-Environment RL Training with Harbor

A unified pipeline for training agents on **mixed datasets** — SWE-bench, Terminal-Bench, custom tasks, etc. — through a single endpoint. Uses **TITO (Token In Token Out)** through SGLang's `/v1/chat/completions` for exact token-level training signals.

**Agent orchestration and grading** are handled by [Harbor](https://github.com/harbor-framework/harbor). Harbor provides unified rollout + grading in a single `Trial.run()` call. The server is **task-type agnostic** — all differentiation (environment, grading harness) is encoded in each task's 4 files (instruction.md, Dockerfile, test.sh, task.toml).

## Architecture

```
Docker Network (swe-net)
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│ Miles container (GPU)              │       │ Harbor container (agent-env)       │
│                                    │       │                                    │
│  agentic_tool_call.generate        │       │  server.py (port 11000)            │
│  (miles/rollout/generate_hub/)     │       │    Wraps Harbor Trial API          │
│    1. Create session on Router     │       │    Task-type agnostic              │
│    2. Call agent_function.run ─────────────►│  3. Harbor Trial.run():            │
│    5. Collect records              │       │     a. Start Docker (task image)   │
│    6. Build training samples       │       │     b. Install agent               │
│    7. Merge agent metadata          │       │     c. Run agent loop ────────┐   │
│    8. merge_samples() multi-turn   │       │                               │   │
│                                    │       │                               │   │
│  Miles Router (port 30000)         │       │                               │   │
│    /sessions/{id}/v1/chat/         │       │                               │   │
│      completions                   │       │                               │   │
│    - Proxies to SGLang             │◄──────────── LLM via OPENAI_API_BASE │   │
│    - Records request + response    │       │     d. Run verifier (test.sh)     │
│      with TITO data                │       │     e. Return TrialResult         │
│                                    │       │                                    │
│  SGLang (port 10090, GPU)          │       │  4. Returns:                       │
│    /v1/chat/completions            │       │     reward, exit_status,           │
│    - Applies chat template         │       │     agent_metrics, eval_report     │
│    - Returns input_token_ids       │       │                                    │
│    - Returns token_id per logprob  │       │  Harbor spawns Docker containers   │
│                                    │       │  from task images (on swe-net)     │
│  Megatron (training)               │       │                                    │
└───────────────────────────────────┘       └───────────────────────────────────┘
```

### Layering

```
┌─────────────────────────────────────────────────────────┐
│  Framework layer (miles/rollout/generate_hub/)           │
│    agentic_tool_call.generate  — TITO tracing, sample   │
│      building, metadata merge, multi-turn merge          │
│    Agent fn returns dict|None — merged into metadata     │
├─────────────────────────────────────────────────────────┤
│  Agent function (examples/.../swe_agent_function.py)      │
│    Task-type agnostic: POST to Harbor server,            │
│    return env info as dict                               │
├─────────────────────────────────────────────────────────┤
│  Env-specific components (examples/.../generate.py)      │
│    reward_func, dynamic_filter, RolloutFn, metrics       │
│    (reads metadata — works for any task type)            │
└─────────────────────────────────────────────────────────┘
```

## How It Works

1. **Session creation**: `agentic_tool_call.generate` creates an `OpenAIEndpointTracer` session on Miles Router
2. **Dispatch to agent**: calls `swe_agent_function.run(base_url, prompt, request_kwargs, metadata)` which POSTs to `server.py`
3. **Harbor Trial**: `server.py` creates a `TrialConfig` from the task directory and runs `Trial.run()`:
   - Starts a Docker container from the task's Dockerfile
   - Installs and runs the agent (determined by `agent_name` in metadata)
   - Runs the verifier (task's `test.sh`)
   - Returns `TrialResult` with reward and metrics
4. **TITO recording**: The agent calls `{base_url}/v1/chat/completions`. Miles Router proxies to SGLang and records each request/response with exact token IDs and logprobs
5. **Sample building**: Records are converted to training `Sample`s with token IDs, logprobs, loss masks
6. **Multi-turn merge**: `merge_samples()` concatenates turns; observation tokens get `loss_mask=0`

## Task-Type Agnosticism

Harbor treats all tasks identically — it reads the 4 files from each task directory:

| File | Purpose | SWE-Bench | Terminal-Bench | Custom |
|------|---------|-----------|----------------|--------|
| `instruction.md` | Agent prompt | Bug description | Terminal task | Anything |
| `task.toml` | Config (timeout, resources) | 3000s timeout | Per-task | Per-task |
| `environment/Dockerfile` | Container image | SWE-bench image | Ubuntu + tools | Custom |
| `tests/test.sh` | Grading logic | swebench.harness.grading | Exit code check | pytest / LLM judge / custom |

The server, agent function, and generate layer have **zero task-type-specific logic**. Routing is done by:
- `instance_id` in metadata → selects the task directory (which contains the right Dockerfile + test.sh)
- `agent_name` in metadata → selects the Harbor agent (mini-swe-agent, terminus-2, etc.)

## Files

| File | Description |
| --- | --- |
| `swe_agent_function.py` | Custom agent function — dispatches to Harbor server, returns env metadata dict (task-type agnostic) |
| `generate.py` | Reward function, dynamic filter, agent metrics aggregation, `RolloutFn` |
| `server.py` | FastAPI server wrapping Harbor Trial API — task-type agnostic (deploy in agent-env container) |
| `run.py` | Training launcher (`--mode train` or `--mode debug`) |
| `download_and_process_data.py` | Download from HuggingFace or local JSONL, convert to Miles format |
| `prepare_harbor_tasks.py` | Convert Miles JSONL to Harbor task directories (generic fallback) |

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_SERVER_URL` | `http://localhost:11000` | Harbor server URL (falls back to `SWE_AGENT_URL`) |
| `AGENT_MODEL_NAME` | `model` | Model name for hosted_vllm (falls back to `SWE_AGENT_MODEL_NAME`) |
| `AGENT_MAX_CONCURRENT` | `8` | Max concurrent trials (falls back to `SWE_AGENT_MAX_CONCURRENT`) |
| `HARBOR_TASKS_DIR` | `/root/harbor_tasks` | Root directory containing task subdirectories |
| `MILES_ROUTER_EXTERNAL_HOST` | (none) | Hostname for Docker containers to reach Miles Router |
| `AGENT_MAX_INPUT_TOKENS` | `32768` | Max input tokens for hosted_vllm model_info (server.py) |
| `AGENT_MAX_OUTPUT_TOKENS` | `8192` | Max output tokens for hosted_vllm model_info (server.py) |

All new env vars fall back to their legacy `SWE_AGENT_*` equivalents for backward compatibility.

## Data Pipeline

```
download_and_process_data.py --input SWE-Gym/SWE-Gym --output swe.jsonl
download_and_process_data.py --input /data/tb.jsonl   --output tb.jsonl  --agent-name terminus-2 --prompt-key instruction
download_and_process_data.py --input /data/my.jsonl   --output my.jsonl  --agent-name my-agent   --prompt-key task_description
        │
        │  Harbor task directories (all go into HARBOR_TASKS_DIR):
        │
        ├──► Harbor official adapter for SWE-bench task dirs (grading parity)
        ├──► harbor run -d terminal-bench@2.0 ... (or Harbor TB adapter)
        ├──► prepare_harbor_tasks.py --input my.jsonl --output /root/harbor_tasks/
        │       Generic fallback for custom data (reads docker_image, test_script from metadata)
        │
        ├──► cat swe.jsonl tb.jsonl my.jsonl > mixed.jsonl
        │
        └──► run.sh --prompt-data mixed.jsonl
                metadata.agent_name tells server.py which Harbor agent to use per task
                metadata.instance_id tells server.py which task directory to use
```

## How to Run

### 1. Create Docker network

```bash
docker network create swe-net
```

### 2. Start containers

**Miles container** (GPU, training + inference):

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --network swe-net \
  --name miles \
  radixark/miles:latest \
  /bin/zsh
```

**agent-env container** (Harbor server, agent execution, Docker-in-Docker):

```bash
docker run -itd \
  --name agent_env \
  --shm-size 16g \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /data:/data \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network swe-net \
  ubuntu:latest \
  /bin/bash
```

### 3. Prepare data and task directories

```bash
# Download and convert to Miles format
python download_and_process_data.py --input SWE-Gym/SWE-Gym --output swe.jsonl
python download_and_process_data.py --input /data/tb.jsonl   --output tb.jsonl  --agent-name terminus-2 --prompt-key instruction
python download_and_process_data.py --input /data/my.jsonl   --output my.jsonl  --agent-name my-agent   --prompt-key task_description

# Merge into one mixed JSONL
cat swe.jsonl tb.jsonl my.jsonl > mixed.jsonl

# Create Harbor task dirs
# For SWE-bench: use Harbor's official adapter (grading parity)
# For Terminal-Bench: harbor run -d terminal-bench@2.0
# For custom data: use prepare_harbor_tasks.py
python prepare_harbor_tasks.py --input my.jsonl --output /root/harbor_tasks/ --docker-network swe-net
```

### 4. Start the Harbor server

```bash
python server.py --port 11000 --max-concurrent 8
```

### 5. Launch training

```bash
python run.py --prompt-data /root/mixed.jsonl \
  --agent-server-url http://agent_env:11000 \
  --router-external-host miles

# Debug mode (smaller batch/rollout for pipeline verification)
python run.py --mode debug --prompt-data /root/mixed.jsonl
```

### Quick Test (single instance)

```bash
curl -X POST http://agent_env:11000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "base_url": "http://miles:30000/sessions/test-session/v1",
    "model": "hosted_vllm/model",
    "instance_id": "django__django-13741",
    "api_key": "dummy"
  }'
```

## Train/Eval Alignment

Using Harbor with the appropriate agent ensures alignment per task type:

| Task Type | Agent | Grading | Docker Images |
| --- | --- | --- | --- |
| SWE-Bench | mini-swe-agent v2 | swebench.harness.grading | Official SWE-bench images |
| Terminal-Bench | terminus-2 | task-specific test.sh | Terminal-Bench images |
| Custom | configurable | custom test.sh | custom Dockerfile |

## TITO Details

- **Token In**: SGLang applies `apply_chat_template()` and tokenizes. The exact `input_token_ids` are returned in the response.
- **Token Out**: Each output logprob item includes `token_id` (the exact token ID).

### Multi-turn merge

```
Turn 1: [prompt_tokens] [response_tokens_1]         -> Sample 0
Turn 2: [prompt_tokens] [response_tokens_1] [observation_tokens] [response_tokens_2] -> Sample 1

merge_samples() ->
  tokens:    [prompt] [resp1] [obs] [resp2]
  loss_mask: -------- [1]*r1  [0]*o [1]*r2
  logprobs:  -------- [real]  [0.0] [real]
```

## Troubleshooting

### Harbor Docker containers can't reach Miles Router

Ensure task directories were created with `--docker-network swe-net` (via `prepare_harbor_tasks.py` or Harbor adapters). This generates a `docker-compose.yaml` override that joins the container to the shared network. Harbor containers must be on the same Docker network as the Miles container.

### `TaskNotFound` error

Ensure the task directory exists under `HARBOR_TASKS_DIR`. Run the appropriate adapter or `prepare_harbor_tasks.py` first.

### First run is slow

Each task uses a Docker image. First pull can be slow; subsequent runs use cached images. Agent installation per container (~30s overhead) can be eliminated by pre-baking into custom images.
