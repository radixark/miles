# SWE-Agent V2: Direct Miles <-> mini-swe-agent with TITO

Replaces the v1 nemo-gym middle layer with direct communication between Miles and mini-swe-agent, using **TITO (Token In Token Out)** through SGLang's `/v1/chat/completions` endpoint for exact token-level training signals.

**Agent orchestration and grading** are handled by [Harbor](https://github.com/harbor-framework/harbor), the official Terminal-Bench 2.0 harness that also supports SWE-bench. Harbor provides unified rollout + grading in a single `Trial.run()` call, replacing the previous `swegym_runner` dependency.

## Architecture

```
Docker Network (swe-net)
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│ Miles container (GPU)              │       │ Harbor container (swe-env)         │
│                                    │       │                                    │
│  agentic_tool_call.generate        │       │  server.py (port 11000)            │
│  (miles/rollout/generate_hub/)     │       │    Wraps Harbor Trial API          │
│    1. Create session on Router     │       │                                    │
│    2. Call swe_agent_function.run ─────────►│  3. Harbor Trial.run():            │
│    5. Collect records              │       │     a. Start Docker (SWE image)   │
│    6. Build training samples       │       │     b. Install mini-swe-agent     │
│    7. Merge agent metadata          │       │     c. Run agent loop ────────┐   │
│    8. merge_samples() multi-turn   │       │                               │   │
│                                    │       │                               │   │
│  Miles Router (port 30000)         │       │                               │   │
│    /sessions/{id}/v1/chat/         │       │                               │   │
│      completions                   │       │                               │   │
│    - Proxies to SGLang             │◄──────────── LLM via OPENAI_API_BASE │   │
│    - Records request + response    │       │     d. Run verifier (pytest)      │
│      with TITO data                │       │     e. Return TrialResult         │
│                                    │       │                                    │
│  SGLang (port 10090, GPU)          │       │  4. Returns:                       │
│    /v1/chat/completions            │       │     reward, exit_status,           │
│    - Applies chat template         │       │     agent_metrics, eval_report     │
│    - Returns input_token_ids       │       │                                    │
│    - Returns token_id per logprob  │       │  Harbor spawns Docker containers   │
│                                    │       │  from SWE-bench images (on swe-net)│
│  Megatron (training)               │       │                                    │
└───────────────────────────────────┘       └───────────────────────────────────┘
```

### Key: Harbor handles agent orchestration + grading. Miles Router sessions handle TITO training data.

### Layering

```
┌─────────────────────────────────────────────────────────┐
│  Framework layer (miles/rollout/generate_hub/)           │
│    agentic_tool_call.generate  — TITO tracing, sample   │
│      building, metadata merge, multi-turn merge          │
│    Agent fn returns dict|None — merged into metadata     │
├─────────────────────────────────────────────────────────┤
│  Agent function (examples/.../swe_agent_function.py)     │
│    Encapsulates all SWE-Agent / Harbor specifics:        │
│    POST to Harbor server, return env info as dict         │
├─────────────────────────────────────────────────────────┤
│  Env-specific components (examples/.../generate.py)      │
│    reward_func, dynamic_filter, RolloutFn, metrics       │
└─────────────────────────────────────────────────────────┘
```

## How Harbor Works

1. **Agent execution**: Harbor starts a Docker container from the SWE-bench image, installs mini-swe-agent, runs it with `OPENAI_API_BASE` pointing to the Miles Router session endpoint
2. **Grading**: After the agent finishes, Harbor runs `test.sh` in the same container using `swebench.harness.grading` to check FAIL_TO_PASS / PASS_TO_PASS tests
3. **Result**: `TrialResult` contains both agent trajectory and verifier reward — one call, no separate eval step

## Train/Eval Alignment

Using Harbor with default mini-swe-agent v2 ensures perfect alignment:


| Aspect         | Training (Harbor + Miles)               | Eval (official SWE-bench) |
| -------------- | --------------------------------------- | ------------------------- |
| Agent          | mini-swe-agent v2                       | mini-swe-agent v2         |
| Format         | Tool-calling (BASH_TOOL)                | Tool-calling (BASH_TOOL)  |
| Config         | Default `swebench.yaml`                 | Default `swebench.yaml`   |
| Docker images  | Official SWE-bench images               | Same                      |
| Submit keyword | `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` | Same                      |
| Grading        | `swebench.harness.grading`              | Same                      |


## Prerequisites

### 1. Miles SGLang patch (TITO)

The Miles Docker image must include the SGLang TITO patch (`docker/patch/v0.5.7/sglang.patch`), which adds:

- `token_id: int` to `ChatCompletionTokenLogprob` (output token IDs)
- `input_token_ids: List[int]` to `ChatCompletionResponseChoice` (input token IDs)
- `logprob_start_len: Optional[int]` to `ChatCompletionRequest` (controls logprob range)
- `input_ids` support in `ChatCompletionRequest` (direct token input)

### 2. PR #656 (Miles-side TITO support)

[PR #656](https://github.com/radixark/miles/pull/656) simplifies `sessions.py` to work with TITO. This PR must be merged before running V2.

### 3. Harbor

Install in the swe-env container:

```bash
pip install harbor-framework
```

## Files


| File                           | Description                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------ |
| `swe_agent_function.py`        | Custom agent function — dispatches to Harbor server, returns env metadata dict |
| `generate.py`                  | Reward function, dynamic filter, agent metrics aggregation, `RolloutFn`        |
| `server.py`                    | FastAPI server wrapping Harbor Trial API (deploy in swe-env container)         |
| `run.sh`                       | Training launch script (Miles side)                                            |
| `download_and_process_data.py` | Download SWE-bench from HuggingFace and convert to Miles JSONL format          |
| `prepare_harbor_tasks.py`      | Convert Miles JSONL to Harbor task directories (one-time)                      |


The generate function itself is `miles.rollout.generate_hub.agentic_tool_call.generate` (framework layer), not in this directory. It is a generic agentic generate that works with any agent/env — the agent function returns a `dict | None` that gets merged into sample metadata for downstream reward models.

## Data Pipeline

```
HuggingFace (SWE-Gym/SWE-Gym)
        │
        ▼
download_and_process_data.py      # Step 1: Download + convert
        │
        ▼
/root/swe_train.jsonl             # Miles JSONL: {"prompt": "...", "metadata": {"instance_id": "...", ...}}
        │
        ├──► prepare_harbor_tasks.py  # Step 2: Create Harbor task dirs (one-time, in swe-env)
        │           │
        │           ▼
        │    /root/harbor_tasks/swebench/
        │    ├── django__django-13741/
        │    │   ├── instruction.md
        │    │   ├── task.toml
        │    │   ├── environment/Dockerfile
        │    │   ├── tests/test.sh
        │    │   └── tests/config.json
        │    └── ...
        │
        └──► run.sh (--prompt-data)   # Step 3: Training uses same JSONL for prompt sampling
```

`download_and_process_data.py` accepts either a HuggingFace dataset path or a local JSONL file. It wraps each instance into Miles format (`{"prompt": ..., "metadata": {...}}`). All original SWE-bench fields (instance_id, repo, problem_statement, test_patch, FAIL_TO_PASS, etc.) are preserved in metadata, which `prepare_harbor_tasks.py` reads to build Harbor task directories.

## How to Run Everything

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

**swe-env container** (Harbor server, agent execution, Docker-in-Docker):

```bash
docker run -itd \
  --name swe_env \
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

> `-v /var/run/docker.sock:/var/run/docker.sock` is required — Harbor spawns SWE-bench Docker containers inside this container.

### 3. Install dependencies in swe-env

```bash
docker exec -it swe_env /bin/bash

# System deps
apt update && apt install -y curl git python3 python3-pip docker.io

# Harbor
pip install harbor-framework

# Verify
python3 -c "from harbor.trial.trial import Trial; print('Harbor OK')"
```

### 4. Download and prepare data

**In the Miles container** — download from HuggingFace:

```bash
docker exec -it miles /bin/zsh
cd /root/miles/examples/experimental/swe-agent-v2

# Option A: Download from HuggingFace
python download_and_process_data.py \
  --input SWE-Gym/SWE-Gym \
  --output /root/swe_train.jsonl

# Option B: Convert a local JSONL file
python download_and_process_data.py \
  --input /data/my_swebench_instances.jsonl \
  --output /root/swe_train.jsonl

# Option C: Limit to N instances (useful for testing)
python download_and_process_data.py \
  --input SWE-Gym/SWE-Gym \
  --output /root/swe_train.jsonl \
  --limit 50
```

**Copy the JSONL to swe-env** (if not using a shared volume):

```bash
docker cp miles:/root/swe_train.jsonl swe_env:/root/swe_train.jsonl
```

### 5. Prepare Harbor task directories (one-time)

**In the swe-env container**:

```bash
docker exec -it swe_env /bin/bash

# Copy scripts from Miles repo (or mount the repo)
# If using shared /data volume:
cp /data/miles/examples/experimental/swe-agent-v2/prepare_harbor_tasks.py /root/
cp /data/miles/examples/experimental/swe-agent-v2/server.py /root/

# Create Harbor task directories from the Miles JSONL
python3 /root/prepare_harbor_tasks.py \
  --input /root/swe_train.jsonl \
  --output /root/harbor_tasks/swebench/

# Verify
ls /root/harbor_tasks/swebench/ | head -5
# Should show instance_id directories like:
# django__django-13741
# scikit-learn__scikit-learn-12345
# ...
```

This reads the Miles JSONL (produced by `download_and_process_data.py`) and creates one Harbor task directory per instance containing the problem statement, Docker config, test harness, and grading scripts.

### 6. Convert reference model (one-time)

**In the Miles container** — convert HF checkpoint to torch_dist format:

```bash
python /root/miles/tools/convert_hf_to_torch_dist.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output /root/qwen3-4B-Instruct-2507_torch_dist \
  --tp 2
```

### 7. Start the Harbor server

**In the swe-env container**:

```bash
docker exec -it swe_env /bin/bash

python3 /root/server.py --port 11000 --max-concurrent 8
```

Keep this running. You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:11000
```

### 8. Verify the server (optional)

**From the Miles container**, test that the Harbor server is reachable:

```bash
# Health check
curl http://swe_env:11000/health
# {"status":"ok"}
```

### 9. Launch training

**In the Miles container**:

```bash
cd /root/miles

# Required env vars
export SWE_AGENT_URL=http://swe_env:11000
export MILES_ROUTER_EXTERNAL_HOST=miles   # so swe-env can reach Miles Router

# Harbor config (defaults are usually fine)
export HARBOR_TASKS_DIR=/root/harbor_tasks/swebench
export HARBOR_DOCKER_NETWORK=swe-net

# Optional: WandB logging
export WANDB_KEY=your_key_here

# Launch!
bash examples/experimental/swe-agent-v2/run.sh
```

This will:

1. Start a Ray cluster with 4 GPUs
2. Launch SGLang inference engines
3. Start Miles Router on port 30000
4. Begin GRPO training loop — each rollout sends instances to the Harbor server, which runs mini-swe-agent in Docker containers, grades results, and returns rewards

### Quick Test (single instance)

Before launching full training, test one instance end-to-end:

```bash
# In Miles container, with SGLang + Miles Router already running:
curl -X POST http://swe_env:11000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "base_url": "http://miles:30000/sessions/test-session/v1",
    "model": "hosted_vllm/model",
    "instance_id": "django__django-13741",
    "api_key": "dummy"
  }'
```

Expected response:

```json
{"reward": 0.0, "exit_status": "Submitted", "agent_metrics": {...}, "eval_report": {...}}
```

## How It Works (Detailed)

1. **Session creation**: `agentic_tool_call.generate` (framework layer) creates an `OpenAIEndpointTracer` session on Miles Router. This gives a `base_url` like `http://miles:30000/sessions/{session_id}`.
2. **Dispatch to agent**: `agentic_tool_call.generate` calls `swe_agent_function.run(base_url, prompt, request_kwargs, metadata)`. The agent function POSTs to `server.py` with `base_url`, `model`, `sampling_params`, and SWE-Gym instance metadata (from `metadata`).
3. **Harbor Trial**: `server.py` creates a `TrialConfig` with the Harbor task path, agent config (mini-swe-agent with `OPENAI_API_BASE` pointing to the session URL), and Docker environment config. `Trial.run()` handles everything:
  - Starts a Docker container from the SWE-bench image
  - Installs and runs mini-swe-agent
  - Runs the verifier (pytest + swebench grading)
  - Returns `TrialResult` with reward and metrics
4. **TITO recording**: mini-swe-agent calls `{base_url}/v1/chat/completions`. Miles Router proxies to SGLang and records each request/response as a `SessionRecord` with exact token IDs and logprobs.
5. **Agent metadata**: `swe_agent_function.run` returns a dict with env info (reward, exit_status, eval_report, agent_metrics). The generate layer merges this into every sample's metadata.
6. **Record collection**: `agentic_tool_call.generate` calls `tracer.collect_records()` to retrieve all session records.
7. **Sample building**: `compute_samples_from_openai_records()` converts each record into a `Sample` with token IDs, logprobs, loss masks, and status (from inference engine finish_reason).
8. **Multi-turn merge**: `merge_samples()` concatenates samples. Observation tokens (user messages, tool outputs between turns) get `loss_mask=0` and `logprob=0.0`, so the model is only trained on its own responses.
9. **Reward**: The outer `generate_and_rm` layer calls `async_rm` / `--custom-rm-path` to compute reward from sample metadata (where the agent left env info).

## Network Setup

- **miles container**: `swe-net` (e.g., 172.18.0.2) — Miles Router on port 30000
- **swe-env container**: `swe-net` (e.g., 172.18.0.3) — Harbor server on port 11000
- **Harbor Docker containers**: Must be on `swe-net` to reach Miles Router

Harbor's Docker containers are configured with `--network swe-net` via the `HARBOR_DOCKER_NETWORK` env var / `EnvironmentConfig.docker_network`.

`OPENAI_API_BASE=http://miles:30000/sessions/{id}/v1` is passed from server.py to Harbor's agent config, which forwards it to the Docker container environment.

## Verification

1. **Harbor install**: `python -c "from harbor.trial.trial import Trial"` in swe-env
2. **Task prep**: `ls /root/harbor_tasks/swebench/ | head` — verify directories exist
3. **Single trial**: `curl -X POST http://localhost:11000/run -d '{"base_url": "http://miles:30000/sessions/test/v1", "model": "hosted_vllm/model", "instance_id": "django__django-13741"}'`
4. **Token format**: Verify session records contain TITO fields
5. **Full training**: Launch run.sh, verify rollouts with non-zero rewards
6. **Comparison**: Reward distribution should be similar to previous swegym_runner baseline

## Troubleshooting

### SGLang missing TITO fields (`input_token_ids`, `token_id`)

The SGLang TITO patch is in `docker/patch/v0.5.7/sglang.patch`. Manually apply if needed:

```bash
cd /sgl-workspace/sglang
git apply /root/miles/docker/patch/v0.5.7/sglang.patch
```

### Harbor Docker containers can't reach Miles Router

Ensure `HARBOR_DOCKER_NETWORK=swe-net` is set. Harbor containers must join the same Docker network as the Miles container.

### `TaskNotFound` error

Run `prepare_harbor_tasks.py` first to create task directories:

```bash
python prepare_harbor_tasks.py --input /root/swe_train.jsonl --output /root/harbor_tasks/swebench/
```

### First run is slow

Each SWE-bench instance uses a Docker image. First pull can be slow; subsequent runs use cached images. mini-swe-agent is installed per container (~30s overhead). This can be eliminated by pre-baking it into custom Docker images.

## Terminal-Bench (Future Phase 2)

Once SWE-bench is working with Harbor:

1. Prepare Terminal-Bench tasks using Harbor adapter
2. Write a `terminal_bench_function.py` that returns a dict with env info — same contract as `swe_agent_function.py`
3. Reuse `agentic_tool_call.generate` as-is — just swap `--custom-agent-function-path`
4. Training data collected via same Miles Router sessions

## TITO Details

**TITO** means:

- **Token In**: SGLang applies `apply_chat_template()` itself and tokenizes the messages. The exact `input_token_ids` it used are returned in the response.
- **Token Out**: Each output logprob item includes `token_id` (the exact token ID), not just the text token. `logprob_start_len=0` makes SGLang compute logprobs from position 0.

### Multi-turn merge

```
Turn 1: [prompt_tokens] [response_tokens_1]         -> Sample 0, loss_mask = [1]*resp1_len
Turn 2: [prompt_tokens] [response_tokens_1] [observation_tokens] [response_tokens_2]
                                                      -> Sample 1, loss_mask = [1]*resp2_len

merge_samples() ->
  tokens:    [prompt] [resp1] [obs] [resp2]
  loss_mask: -------- [1]*r1  [0]*o [1]*r2     (observation tokens masked out)
  logprobs:  -------- [real]  [0.0] [real]
```

Token continuity is guaranteed because each turn's `input_token_ids` extends the previous turn's full token sequence (SGLang re-tokenizes the full message history each turn).