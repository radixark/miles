# SWE-Agent training with Harbor on Docker sandboxes

This example trains agentic coding models with GRPO. Miles runs the training
loop and serves the policy through its session server; a separate
[Harbor](https://github.com/harbor-framework/harbor) agent server creates the
task sandboxes, runs the agents, and returns verifier rewards.

Two recipes share this pipeline: **GLM-4.7-Flash** (synchronous, on NVIDIA) and
**Qwen3-Coder-30B-A3B** (colocate or async, tested on 8× AMD MI350X). Both support
Terminal-Bench, SWE-bench, and custom Harbor tasks. Training records must
contain a `prompt` and `metadata.instance_id` identifying the Harbor task.

## Files

| File | Purpose |
| --- | --- |
| `run.py` | Validated synchronous GLM-4.7-Flash launcher. |
| `run-glm47-flash-agentic-async.py` | Disaggregated fully asynchronous GLM launcher. |
| `run-qwen3-swe.py` | Qwen3-Coder-30B-A3B launcher (colocate or async, tested on 8× AMD MI350X). |
| `run_glm52_lora_tb2_daytona.py` | Multi-node GLM-5.2 744B-A40B LoRA launcher (bf16 trainer, fp8 rollout). |
| `swe_agent_function.py` | Sends each rollout to the Harbor agent server. |
| `generate.py` | Builds rewards, metrics, and training samples. |
| `download_and_process_data.py` | Converts supported datasets to Miles JSONL. |

## 1. Start the Harbor agent server

This step is shared by both recipes. After it, complete exactly one training
recipe — **2(a)** (GLM-4.7-Flash) or **2(b)** (Qwen3-Coder-30B) — then follow
the common **3. Networking** and **4. Verify progress** sections.

Use the `harbor-miles-v0.20.0` branch of the `harbor-framework/harbor`
repository, which carries the Miles integration:

```bash
git clone https://github.com/harbor-framework/harbor.git
cd harbor
git checkout harbor-miles-v0.20.0
uv sync

HARBOR_TASKS_DIR=/path/to/harbor_tasks uv run python miles_agent_server.py \
    --host 0.0.0.0 \
    --port 30000 \
    --dashboard-port 0 \
    --max-concurrent 32 \
    --agent-timeout 5400 \
    --trials-dir /path/to/trials
```

`HARBOR_TASKS_DIR` must contain one Harbor task directory for every
`metadata.instance_id` in the training data. The agent-server machine must have
Docker and enough capacity for the requested number of concurrent sandboxes;
set `--max-concurrent` to at least one sandbox per trajectory in a rollout step
(`--rollout-batch-size` times `--n-samples-per-prompt`). Keep `--agent-timeout`
generous — agentic trials routinely run past an hour, and a short timeout kills
them mid-episode. Verify `http://<agent-server>:30000/health` before launching
Miles.

The two per-trial timeouts must be ordered. `--agent-timeout` is the authoritative
one: when it fires, the agent server ends the trial and frees its sandbox. The
rollout client applies a second ceiling, `AGENT_TRIAL_TIMEOUT` (default 7200
seconds), which has to stay above `--agent-timeout`. If the client gives up first,
the trial is recorded as aborted while the agent server keeps running it, so the
sandbox and its `--max-concurrent` slot stay busy for the remaining difference, and
the aborted sample takes its whole GRPO group down with it. Raise it through the
launcher's generic env-var hook:

```bash
python examples/swe-agent-harbor-docker/run.py ... --extra-env-vars 'AGENT_TRIAL_TIMEOUT=10800'
```

## 2(a). GLM-4.7-Flash (synchronous)

### Prepare Terminal-Bench data

Convert a local JSONL whose rows include a task instruction and instance name:

```bash
python examples/swe-agent-harbor-docker/download_and_process_data.py \
    --input /path/to/terminal-bench.jsonl \
    --output /path/to/tb2_train.jsonl \
    --agent-name mini-swe-agent \
    --prompt-key instruction
```

The resulting `metadata.instance_id` values must match task directories known to
the Harbor agent server.

### Launch

The shape below is what a multi-day Terminal-Bench 2 run used on one node of 8
H200 GPUs: 32 trajectories per GRPO step (4 prompts times 8 samples), each one a
full mini-swe-agent episode in its own Harbor sandbox.

```bash
python examples/swe-agent-harbor-docker/run.py \
    --num-nodes 1 \
    --num-gpus-per-node 8 \
    --skip-prepare \
    --megatron-path /root/Megatron-LM \
    --hf-checkpoint /path/to/GLM-4.7-Flash \
    --ref-load /path/to/GLM-4.7-Flash_torch_dist \
    --save-dir /path/to/checkpoints \
    --prompt-data /path/to/tb2_train.jsonl \
    --max-seq-len 65536 \
    --rollout-batch-size 4 \
    --n-samples-per-prompt 8 \
    --global-batch-size 32 \
    --num-rollout 200 \
    --save-interval 20 \
    --agent-server-url http://<agent-server>:30000 \
    --router-external-host <trainer-host-reachable-from-agent-server> \
    --miles-host-ip 0.0.0.0 \
    --save-traces-dir /path/to/traces
```

Expect roughly 10 minutes per step at this shape; because synchronous rollout
waits for the slowest trajectory in the batch, a step that draws an unusually
slow task can take several times that. The synchronous launcher uses GLM-4.7
tool-call and reasoning parsers, TITO, the Miles session server, and the
Megatron backend.

## 2(b). Qwen3-Coder-30B-A3B (tested on 8× AMD MI350X)

Runs **colocate** (default: `train.py`, TP=1, EP=8, all 8 GPUs shared) or
**async** (`--async-mode`: `train_async.py`, TP=2, EP=4, split 4 train + 4
rollout via `--train-num-gpus`).

### Prepare SWE-Gym data

Build the Harbor task directories with the SWE-Gym adapter, then convert the same
set to Miles JSONL. `--dataset lite` is SWE-Gym-Lite (230 tasks) and `--dataset
full` is SWE-Gym (2438 tasks); both sides must reference the same set so the
`instance_id`s line up.

```bash
# Harbor task directories (run in the harbor repo)
cd ~/harbor/adapters/swegym
uv run --frozen --no-dev --with 'swebench==4.1.0' run_adapter.py \
    --dataset lite \
    --task-dir /data/miles_ci/harbor_tasks \
    --overwrite

# Training JSONL (same SWE-Gym-Lite set)
python examples/swe-agent-harbor-docker/download_and_process_data.py \
    --input SWE-Gym/SWE-Gym-Lite \
    --output /root/datasets/swe_gym_lite.jsonl \
    --agent-name mini-swe-agent \
    --prompt-key problem_statement
```

`--dataset lite` maps to the `SWE-Gym/SWE-Gym-Lite` HuggingFace dataset — the
same id the JSONL step converts — so `metadata.instance_id` matches the task
directories the agent server serves. Pair `--dataset full` with `--input
SWE-Gym/SWE-Gym`.

### Launch

```bash
python examples/swe-agent-harbor-docker/run-qwen3-swe.py \
    --async-mode \
    --skip-prepare \
    --num-gpus-per-node 8 \
    --train-num-gpus 4 \
    --megatron-path /root/Megatron-LM \
    --hf-checkpoint /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --ref-load /path/to/Qwen3-Coder-30B-A3B-Instruct_torch_dist \
    --save-dir /path/to/checkpoints \
    --prompt-data /root/datasets/swe_gym_lite.jsonl \
    --harbor-tasks-dir /data/miles_ci/harbor_tasks \
    --agent-server-url http://<agent-server>:30000 \
    --router-external-host <trainer-host-reachable-from-agent-server> \
    --miles-trials-dir /data/miles_ci/trials \
    --num-rollout 15 --save-interval 15
```

Drop `--skip-prepare` on the first run to convert the HF checkpoint to
torch_dist. `--ref-load`'s basename must be `<model>_torch_dist`, since `prepare`
writes the conversion into its parent directory. The launcher uses Qwen3
(`qwen25` / `qwen3`) tool-call and reasoning parsers, TITO, the Miles session
server, and the Megatron backend, and on AMD ROCm the Triton MoE backend.

## 3. Networking

For a smoke test on either recipe, set `--num-rollout 1`.

`--router-external-host` is the address Harbor sandboxes use to call the Miles
session server and SGLang router. It must resolve and route from the agent-server
machine. `--miles-host-ip 0.0.0.0` is useful when those services must accept
connections forwarded from another host. The launcher starts 32 session-server
workers on ports 30000-30031 and the SGLang router on port 31000, so ensure that
range and port are reachable end to end; Tailscale is one option when the
machines are on different networks. If the trainer reaches the agent server
through a proxy or an in-cluster service rather than directly, point
`--agent-server-url` at that stable name rather than an ephemeral pod address.
The rollout client enables TCP keepalive probes so long-running trials do not
lose an idle connection while Harbor is working.

## 4. Verify progress

Check both layers:

1. Miles logs emit rollout metrics and write `rollout_data/*.pt` under the trace
   directory.
2. Megatron logs emit `train/step` and the Ray job exits successfully.

Confirm a suspected stall on disk before believing a dashboard. W&B uploads can
fail partway through a long run — dropping some metric rows while others keep
arriving — which looks exactly like a frozen reward curve. The per-step
`train_data/<step>` and `rollout_data/<step>.pt` dumps under `--save-traces-dir`
are written by the trainer itself and are the authoritative progress signal.
