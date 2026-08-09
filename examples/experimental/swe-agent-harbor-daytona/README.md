# SWE-Agent training with Harbor on Daytona sandboxes

This example trains GLM-4.7-Flash on agentic terminal and coding tasks, with
task sandboxes hosted on [Daytona](https://www.daytona.io/) instead of local
Docker. Miles runs synchronous GRPO and serves the policy through its session
server; a Harbor agent server drives the agent and returns verifier rewards.
It is meant to run on a single node of 8 H200 GPUs.

It is the `examples/swe-agent-harbor-docker` pipeline with two changes:

- **Daytona sandboxes.** The agent-server host needs outbound HTTPS but no
  Docker daemon, no local image builds, and no local disk for task images. This
  is the practical option when the trainer runs on a GPU node where you cannot
  or do not want to run Docker-in-Docker.
- **terminus-2 agent.** terminus-2 runs as a host process and calls the model
  endpoint itself, rather than from inside the sandbox, so the model endpoint
  must be reachable from the agent-server host.

Everything else — TITO, the session server, GRPO, the reward path — is identical
to `examples/swe-agent-harbor-docker`, and so is the trainer side: this example has no
launcher of its own and runs `examples/swe-agent-harbor-docker/run.py` unchanged. Daytona is
selected entirely by the agent server's environment, which the trainer never
sees.

## Files

| File | Purpose |
| --- | --- |
| `launch_agent_server.sh` | Starts the Harbor agent server in Daytona mode. |

## 1. Provision Daytona

Create an API key in the Daytona dashboard and export it on the agent-server
host:

```bash
export DAYTONA_API_KEY=<your-daytona-api-key>
```

Daytona accounts have a total-disk quota, so keep concurrent sandboxes times
`HARBOR_DAYTONA_DISK_GB` under it.

## 2. Start the Harbor agent server

Use the `harbor-miles-v0.20.0` branch of `harbor-framework/harbor`, which
carries the Miles integration:

```bash
git clone https://github.com/harbor-framework/harbor.git
cd harbor
git checkout harbor-miles-v0.20.0
uv sync

export DAYTONA_API_KEY=<your-daytona-api-key>
export HARBOR_TASKS_DIR=/path/to/harbor_tasks
export TRIALS_DIR=/path/to/trials
bash /path/to/miles/examples/experimental/swe-agent-harbor-daytona/launch_agent_server.sh
```

`HARBOR_TASKS_DIR` must contain one Harbor task directory for every
`metadata.instance_id` in the training data; a missing directory makes the trial
score 0 rather than raise. Set `--max-concurrent` to at least one sandbox per
trajectory in a rollout step (`--rollout-batch-size` times
`--n-samples-per-prompt`). Keep the agent timeout generous — agentic trials
routinely run past an hour.

Run the agent server under a process supervisor or a detached terminal
multiplexer on its own host, not in a foreground shell over SSH: if that shell
dies it takes the agent server and every live sandbox with it, and the trainer
then starves without an obvious error.

Verify `http://<agent-server>:11000/health` before launching Miles.

## 3. Prepare data

`examples/swe-agent-harbor-docker/download_and_process_data.py` converts a local JSONL into
Miles format. For terminus-2, set the agent name accordingly:

```bash
python examples/swe-agent-harbor-docker/download_and_process_data.py \
    --input /path/to/terminal-bench.jsonl \
    --output /path/to/tb2_train.jsonl \
    --agent-name terminus-2 \
    --prompt-key instruction
```

## 4. Launch training

The shape below is what a multi-day Terminal-Bench 2 run used, with the agent
server colocated on the same host as the trainer:

```bash
export WANDB_API_KEY=<your-wandb-key>

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
    --save-interval 10 \
    --agent-server-url http://127.0.0.1:11000 \
    --router-external-host <trainer-address-reachable-from-agent-server> \
    --save-traces-dir /path/to/traces \
    --wandb-project <your-wandb-project>
```

For a smoke test, set `--num-rollout 1`.

`--router-external-host` is the address the agent server uses to reach the Miles
session server, substituted into the base URL handed to the agent. It only has
to resolve from the agent-server host, so a hostname is fine — use one when the
agent server reaches the trainer over a tailnet or other overlay. Do not confuse
it with `--miles-host-ip`, which is bound locally on the trainer and must be an
address that already exists on one of its interfaces. Ports 30000 and 31000 must
be reachable from the agent-server host.

## Sizing the per-turn response cap

`--rollout-max-response-len` and the agent server's `AGENT_MAX_OUTPUT_TOKENS`
both cap a **single turn**, not the whole trajectory. Agentic trajectories are
routinely several times longer than one turn, so a cap that looks generous
against `--max-seq-len` can still abort most trials.

When a turn exceeds the cap, `HARBOR_RESPONSE_LENGTH_POLICY=abort` ends the
trial and **none of that turn's tool calls are performed**, so the trial scores
0 and dilutes its GRPO group. The symptoms are
`SingleTurnMaxSeqLenExceededError` and `ContextLengthExceededError` in the trial
exception files, with `rollout/truncated_ratio` well above 0.

To size these, compare `rollout/response_len/mean` and `rollout/response_len/max`
against the cap, and keep `AGENT_MAX_INPUT_TOKENS` above the largest observed
context.
`--max-seq-len 65536` leaves plenty of headroom to raise both.

`examples/swe-agent-harbor-docker/run.py` hardcodes `--rollout-max-response-len 8192`, so raise
it there; `AGENT_MAX_OUTPUT_TOKENS` is an environment variable on the agent
server and is set in `launch_agent_server.sh`. Raise the two together — leaving
either one behind reintroduces the aborts.

## Verify progress

Read `rollout/raw_reward` for the task solve rate. `rollout/rewards` is the
GRPO-centered advantage and sits near zero by construction, so it never shows
learning.

Two properties of this shape make per-step reward misleading:

- With `--rollout-batch-size 4` there are only 4 GRPO groups per step. Uniform
  groups (all solved or all failed) contribute no gradient, and which 4 tasks
  were drawn dominates the step reward. Judge progress on repeated tasks across
  many batches, never on consecutive steps.
- A long run's headline health number is the fraction of trials that return
  successfully, not the reward. Census the trial directories under
  `--trials-dir` by outcome: no `exception.txt` means success, and the last
  exception class named in that file is the failure mode. Key the census on the
  **trial start time** — the mtime of the trial's `config.json`, written at
  launch — because writing `exception.txt` bumps the directory mtime and makes
  any mtime-sorted listing look like everything is failing.

Confirm a suspected stall on disk before believing a dashboard. W&B uploads can
fail partway through a long run, dropping some metric rows while others keep
arriving, which looks exactly like a frozen reward curve. The per-step
`train_data/<step>` and `rollout_data/<step>.pt` dumps under `--save-traces-dir`
are written by the trainer itself and are authoritative.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `DaytonaValidationError` on sandbox create | Daytona disk quota exhausted. |
| `EnvironmentStartTimeoutError` in bursts | Sandbox creation is slow because the account is near its disk quota. |
| `SingleTurnMaxSeqLenExceededError` | Per-turn output cap too low; see the sizing section. |
| `ContextLengthExceededError` | `AGENT_MAX_INPUT_TOKENS` below the observed context length. |
| sgl-router fails to bind | `--miles-host-ip` is not an address the trainer host can bind; leave it unset to auto-detect. |
| Every trial scores 0 | `metadata.instance_id` values have no matching directory under `HARBOR_TASKS_DIR`. |
