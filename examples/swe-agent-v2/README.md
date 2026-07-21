# SWE-Agent V2 training with Harbor

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
| `swe_agent_function.py` | Sends each rollout to the Harbor agent server. |
| `generate.py` | Builds rewards, metrics, and training samples. |
| `download_and_process_data.py` | Converts supported datasets to Miles JSONL. |

## 1. Start the Harbor agent server

Use the public Harbor repository and its Miles integration branch, not
`harbor-private`:

```bash
git clone https://github.com/harbor-framework/harbor.git
cd harbor
git checkout harbor-miles-v0.13.1
uv sync

HARBOR_TASKS_DIR=/path/to/harbor_tasks uv run python miles_agent_server.py \
    --host 0.0.0.0 \
    --port 30000 \
    --dashboard-port 0 \
    --max-concurrent 8 \
    --agent-timeout 5400 \
    --trials-dir /path/to/trials
```

`HARBOR_TASKS_DIR` must contain one Harbor task directory for every
`metadata.instance_id` in the training data. The agent-server machine must have
Docker and enough capacity for the requested number of concurrent sandboxes.
Verify `http://<agent-server>:30000/health` before launching Miles.

## 2. Prepare Terminal-Bench data

Convert a local JSONL whose rows include a task instruction and instance name:

```bash
python examples/swe-agent-v2/download_and_process_data.py \
    --input /path/to/terminal-bench.jsonl \
    --output /path/to/tb2_train.jsonl \
    --agent-name mini-swe-agent \
    --prompt-key instruction
```

The resulting `metadata.instance_id` values must match task directories known to
the Harbor agent server.

## 3. Launch synchronous GLM-4.7-Flash training

The following one-node shape was validated on 8 H200 GPUs with eight Terminal-
Bench trajectories (two prompts times four samples), followed by a Megatron
training step:

```bash
python examples/swe-agent-v2/run.py \
    --num-nodes 1 \
    --num-gpus-per-node 8 \
    --skip-prepare \
    --megatron-path /root/Megatron-LM \
    --hf-checkpoint /path/to/GLM-4.7-Flash \
    --ref-load /path/to/GLM-4.7-Flash_torch_dist \
    --save-dir /path/to/checkpoints \
    --prompt-data /path/to/tb2_train.jsonl \
    --max-seq-len 65536 \
    --rollout-batch-size 2 \
    --n-samples-per-prompt 4 \
    --global-batch-size 8 \
    --agent-server-url http://<agent-server>:30000 \
    --router-external-host <trainer-host-reachable-from-agent-server> \
    --miles-host-ip 0.0.0.0 \
    --save-traces-dir /path/to/traces
```

For a smoke test, add `--num-rollout 1`. For a long run, leave the default or
set the desired rollout count explicitly.

`--router-external-host` is the address Harbor sandboxes use to call the Miles
session server and SGLang router. It must resolve and route from the agent-server
machine. `--miles-host-ip 0.0.0.0` is useful when those services must accept
connections forwarded from another host. Ensure ports 30000 and 31000 are
reachable end to end; Tailscale is one option when the machines are on different
networks.

## 4. Verify progress

Check all three layers:

1. Harbor trial logs show increasing `mini-swe-agent (step N)` values.
2. Miles logs emit rollout metrics and write `rollout_data/*.pt` under the trace
   directory.
3. Megatron logs emit `train/step` and the Ray job exits successfully.

The synchronous launcher uses GLM-4.7 tool-call and reasoning parsers, TITO,
the Miles session server, and the Megatron backend. The asynchronous launcher is
available for multi-node disaggregated runs but is not the one-node recipe above.
