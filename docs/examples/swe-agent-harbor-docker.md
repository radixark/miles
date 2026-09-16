---
title: "SWE-Agent training with Harbor on Docker sandboxes"
description: "Trains coding and terminal agents with Harbor-managed local Docker sandboxes and verifier rewards."
# Generated from examples/swe-agent-harbor-docker/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
This example trains GLM-4.7-Flash on agentic coding and terminal tasks. Miles
runs synchronous GRPO and serves the policy through its session server; a
separate [Harbor](https://github.com/harbor-framework/harbor) agent server
creates the task sandboxes, runs the agents, and returns verifier rewards.

The same pipeline supports Terminal-Bench, SWE-bench, and custom Harbor tasks.
Training records must contain a `prompt` and `metadata.instance_id` identifying
the Harbor task.

## Files

| File | Purpose |
| --- | --- |
| `run.py` | Validated synchronous GLM-4.7-Flash launcher. |
| `run-glm47-flash-agentic-async.py` | Disaggregated fully asynchronous launcher. |
| `run_glm52_lora_tb2_daytona.py` | Multi-node GLM-5.2 744B-A40B LoRA launcher (bf16 trainer, fp8 rollout). |
| `swe_agent_function.py` | Sends each rollout to the Harbor agent server. |
| `generate.py` | Builds rewards, metrics, and training samples. |
| `download_and_process_data.py` | Converts supported datasets to Miles JSONL. |

## 1. Start the Harbor agent server

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

If the trainer reaches the agent server through a proxy or an in-cluster service
rather than directly, point `--agent-server-url` at that stable name rather than
an ephemeral pod address. The rollout client enables TCP keepalive probes so
long-running trials do not lose an idle connection while Harbor is working.

## 2. Prepare Terminal-Bench data

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

## 3. Launch synchronous GLM-4.7-Flash training

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

For a smoke test, set `--num-rollout 1`. Expect roughly 10 minutes per step at
this shape; because synchronous rollout waits for the slowest trajectory in the
batch, a step that draws an unusually slow task can take several times that.

`--router-external-host` is the address Harbor sandboxes use to call the Miles session server and SGLang router. It must resolve and route from the agent-server machine. `--miles-host-ip 0.0.0.0` is useful when those services must accept connections forwarded from another host. The launcher starts 32 session-server workers on ports 30000-30031 and the SGLang router on port 31000, so ensure that range and port are reachable end to end; Tailscale is one option when the machines are on different networks.

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

The synchronous launcher uses GLM-4.7 tool-call and reasoning parsers, TITO,
the Miles session server, and the Megatron backend.
