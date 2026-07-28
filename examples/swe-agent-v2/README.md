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

When the trainer reaches the machine through the Kubernetes Tailscale egress
service, use its stable service name, for example
`http://egress-agent-server.tailscale.svc.cluster.local:8080`. The rollout
client enables TCP keepalive probes so long-running trials do not lose an idle
connection while Harbor is working.

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

The shape below is what a multi-day Terminal-Bench 2 run used on one node of 8
H200 GPUs: 32 trajectories per GRPO step (4 prompts times 8 samples), each one a
full mini-swe-agent episode in its own Harbor sandbox.

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
this shape once the task pool is free of the stragglers described below; a step
that draws a pathological task can take several times that.

`--router-external-host` is the address Harbor sandboxes use to call the Miles
session server and SGLang router. It must resolve and route from the agent-server
machine. `--miles-host-ip 0.0.0.0` is useful when those services must accept
connections forwarded from another host. Ensure ports 30000 and 31000 are
reachable end to end; Tailscale is one option when the machines are on different
networks.

## 4. Choose the task pool carefully

Two properties of the task pool dominate how useful the run is, and both are
easy to get wrong.

**Step time is set by the slowest trajectory.** Synchronous rollout submits
`--over-sampling-batch-size` groups and waits until `--rollout-batch-size` of
them pass the dynamic sampling filter; leftovers are aborted. Because that flag
defaults to `--rollout-batch-size`, the default configuration submits exactly as
many groups as it needs and therefore has to wait for every one of them. In the
run above, the median trajectory took 3.5 minutes and the 90th percentile 10
minutes, but two of the 17 tasks averaged 13 and 25 minutes — one of them
looping for 300+ agent steps on a brute-force search. Drawing 4 prompts from a
17-task pool hits at least one of those two about 43% of the time, and those
steps took 35-60 minutes against 8-14 for the rest: a ~2.5x wall-clock tax. Drop
tasks that are both slow and rarely solved, or raise
`--over-sampling-batch-size` above `--rollout-batch-size` so a straggler group
can be abandoned.

**Tasks the model always or never solves teach it nothing.** GRPO normalizes
advantages within a group, so a task with a uniform outcome across its
`--n-samples-per-prompt` samples contributes no gradient. The pool above was
selected as tasks the base model had already solved; its measured pass rates
ranged from 0.00 to 0.94 and averaged 0.483, and `rollout/raw_reward` sat at
0.486 for the whole run — the mean of a fixed task mix, not a learning curve.
Prefer tasks the base model solves *sometimes*, and watch
`rollout/zero_std/all_zero_percentage` and `rollout/zero_std/all_one_percentage`
for the fraction of groups that carry no gradient. Swapping the dynamic sampling
filter to `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`
discards those groups outright, at the cost of more rollouts per step.

## 5. Verify progress

Check all three layers:

1. Harbor trial logs show increasing `mini-swe-agent (step N)` values.
2. Miles logs emit rollout metrics and write `rollout_data/*.pt` under the trace
   directory.
3. Megatron logs emit `train/step` and the Ray job exits successfully.

Confirm a suspected stall on disk before believing a dashboard. W&B uploads can
fail partway through a long run — dropping some metric rows while others keep
arriving — which looks exactly like a frozen reward curve. The per-step
`train_data/<step>` and `rollout_data/<step>.pt` dumps under `--save-traces-dir`
are written by the trainer itself and are the authoritative progress signal.

The synchronous launcher uses GLM-4.7 tool-call and reasoning parsers, TITO,
the Miles session server, and the Megatron backend. The asynchronous launcher is
available for multi-node disaggregated runs but is not the one-node recipe above.
