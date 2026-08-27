# Terminus 2 training with compaction

This example trains GLM-4.7-Flash on all 89 Terminal-Bench 2 tasks with the
Terminus 2 harness. Terminus summarization and linear history are enabled, so a
long agent episode can become multiple training samples without counting as
multiple GRPO rollouts.

The default launcher is a one-node, 8-GPU, 100-step recipe:

- 4 prompts per step;
- 8 independent rollouts per prompt (32 rollouts per step);
- 32,768 tokens per trajectory segment;
- 8,192 generated tokens per model turn; and
- Miles dashboard telemetry and rollout traces enabled.

The launcher reuses the Harbor request, reward, and metric adapters in
`examples/swe-agent-harbor-docker`. Only the compaction-specific training recipe
lives here.

## Mental model

A normal Terminus 2 episode produces one trajectory. When the context becomes
large, Terminus can summarize the conversation and continue from the summary.
With `linear_history=true`, Harbor exposes the pre-summary and post-summary
histories as separate linear branches.

Miles session server v2 records those branches as a trajectory tree and returns
one `Sample` for each kept leaf. Its default postprocessor then:

1. assigns every sibling sample the episode's terminal reward;
2. keeps the same rollout ID on all samples from that episode;
3. masks environment tokens; and
4. masks a shared model completion in every sibling except its first owner.

The result is that every policy token contributes to the loss once. GRPO first
normalizes one reward per original rollout within its prompt group, then
broadcasts that advantage to the rollout's compacted samples. An episode does
not get extra reward weight merely because it compacted more often.

## 1. Start the Harbor agent server

Use the `harbor-miles-v0.20.0` branch of
[`harbor-framework/harbor`](https://github.com/harbor-framework/harbor). It
contains the Miles agent server and the Terminus 2 summarization controls used
by this example.

```bash
git clone https://github.com/harbor-framework/harbor.git
cd harbor
git checkout harbor-miles-v0.20.0
uv sync

export HARBOR_TASKS_DIR=/path/to/terminal-bench-2/tasks
export TRIALS_DIR=/path/to/harbor-trials

export HARBOR_TERMINUS_2_ENABLE_SUMMARIZE=true
export HARBOR_TERMINUS_2_LINEAR_HISTORY=true
export HARBOR_TIMEOUT_MULTIPLIER=1.0
export HARBOR_DELETE_CONTAINERS=true
export HARBOR_OVERRIDE_MEMORY_MB=16384
export HARBOR_RESPONSE_LENGTH_POLICY=abort
export AGENT_MAX_INPUT_TOKENS=32768
export AGENT_MAX_OUTPUT_TOKENS=8192

uv run python miles_agent_server.py \
    --host 0.0.0.0 \
    --port 11000 \
    --dashboard-port 0 \
    --max-concurrent 32 \
    --agent-timeout 3600 \
    --trials-dir "$TRIALS_DIR"
```

`HARBOR_TASKS_DIR` must contain all 89 Terminal-Bench 2 task directories. The
server runs Terminus 2 as a host process and creates one Docker sandbox per
trial, so the server host needs Docker, enough memory for 32 concurrent trials,
and network access to the Miles session server.

Verify the server before launching training:

```bash
curl --fail http://<agent-server>:11000/health
```

## 2. Prepare GLM-4.7-Flash and the full task set

By default, `run.py` expects these directories:

```text
/root/models/GLM-4.7-Flash
/root/models/GLM-4.7-Flash_torch_dist
```

Use `--model-dir` to change the common parent. If only the Hugging Face
checkpoint exists, omit `--skip-prepare` and the launcher converts the reference
checkpoint before training.

Convert a JSONL export of all 89 Terminal-Bench 2 tasks to Miles format. Each
source row must contain `instance_id` and `instruction`:

```bash
python examples/swe-agent-harbor-docker/download_and_process_data.py \
    --input /path/to/terminal-bench-2.jsonl \
    --output /path/to/tb2_train_89.jsonl \
    --agent-name terminus-2 \
    --prompt-key instruction

test "$(wc -l < /path/to/tb2_train_89.jsonl)" -eq 89
```

The resulting `metadata.instance_id` values must match the 89 task directory
names under `HARBOR_TASKS_DIR`. Each row uses this schema:

```json
{"prompt":"task-name","metadata":{"instance_id":"task-name","agent_name":"terminus-2"}}
```

## 3. Launch Miles

Run the launcher from a Miles checkout installed according to the
[installation guide](../../../docs/getting-started/installation.md).
Export `WANDB_API_KEY` if W&B logging is wanted; the launcher works without it.
The following command assumes the Harbor server is on another reachable host:

```bash
export WANDB_API_KEY=<your-wandb-key>

python examples/experimental/terminus-compaction/run.py \
    --skip-prepare \
    --model-dir /path/to/models \
    --output-dir /path/to/output \
    --prompt-data /path/to/tb2_train_89.jsonl \
    --agent-server-url http://<agent-server>:11000 \
    --session-server-ip 0.0.0.0 \
    --router-external-host <trainer-address-reachable-from-agent-server>
```

`--session-server-ip` is the local bind address. `--router-external-host` is the address Harbor uses to call back into the session server and SGLang router; it must resolve from the agent-server host. The launcher starts 32 session-server workers on ports 30000-30031 and the SGLang router on port 31000, so allow inbound TCP connections to that range and port from the agent-server host.

For a local smoke test, run Harbor on the trainer, use
`--agent-server-url http://127.0.0.1:11000`, and set both session addresses to a
local address. Use `--num-rollout 1` to stop after one step.

## 4. Verify compaction and training

The trace root defaults to `<output-dir>/<run-id>/details`. View the live Miles
dashboard with:

```bash
python -m miles.dashboard.serve \
    --dump-details /path/to/output/<run-id>/details \
    --follow \
    --port 7788
```

Useful per-step metrics are:

- `rollout/num_training_samples`: flattened samples trained in the step. It is
  32 without compaction and rises when episodes produce extra retained leaves.
- `rollout/episode_raw_reward`: terminal reward averaged once per original
  rollout, so compacted episodes are not over-weighted.
- `rollout/raw_reward`: reward averaged over flattened samples; this can differ
  from the episode-level metric when rollouts produce different sample counts.
- `rollout/truncated_ratio`: should stay low. A high value usually means a
  per-turn or total-context cap is too small.

Inspect `rollout_data/<step>.pt` or the dashboard's sample view to confirm that:

- compacted siblings share a rollout ID and terminal reward;
- shared prefix completions are loss-masked in all but one sibling; and
- every retained sample has at least one trainable token.

If no rollout compacts, the infrastructure can still be healthy: summarization
only triggers when Terminus reaches its context threshold. Longer or harder
tasks make the path easier to exercise.
