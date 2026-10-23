# TMax score-centering baseline

> **Read the docs:** [Harbor](../../../docs/user-guide/harbor.md) and
> [score centering](../../../docs/examples/infra-features/score-centering.md).

This example runs Qwen3.5-9B with GRPO group-mean advantages and Miles'
score-centering loss using top-128 rollout log probabilities. It uses fully
async training with one trainer GPU and three independent rollout engines.
The training defaults are a short infrastructure smoke test, not the paper's
full run. Evaluation uses all 89 `terminal-bench@2.0` tasks.

## Harness and data

`agent.py` adapts the TMax Vanillux training protocol to Harbor and the Miles
TITO session server. It preserves the **released training system and user
messages**, the single bash tool, persistent working directory and exported
environment, output truncation, 120-second command timeout, and
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` submission marker. The released
training system message differs from Vanillux2Agent's evaluation system
message; the data messages take precedence here. Do not enable
`--apply-chat-template`: the session server owns tokenization.

Each task uses its official Docker image and unmodified released verifier.
Harbor uploads tests after the agent finishes. An episode without an observed
submit marker receives zero, even if the final files would pass the verifier.
Infrastructure errors are raised instead of being quietly assigned a reward.
If the API rejects a turn because the accumulated conversation plus the fixed
completion cap exceeds the model context window, the adapter retries that turn
with the largest completion budget that fits; unrelated API errors still fail
the trial.

Sources, pinned to TMax commit `6d3d606c0f9de6ceec9481f9ccc793936818fef0`:

- [Training environment](https://github.com/hamishivi/tmax/blob/6d3d606c0f9de6ceec9481f9ccc793936818fef0/training/open-instruct/open_instruct/environments/swerl_vanillux_sandbox.py)
- [Vanillux2Agent](https://github.com/hamishivi/tmax/blob/6d3d606c0f9de6ceec9481f9ccc793936818fef0/Vanillux2Agent/agent.py)
- [Released training data](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct)

Prepare a smoke subset from the downloaded Parquet and extracted task-data
archive (omit `--task-id` to prepare all tasks):

```bash
python examples/experimental/tmax_score_centering/prepare_data.py \
  --input /data/tmax-open-instruct/data/train-00000-of-00001.parquet \
  --task-dir /data/tmax-task-data \
  --output /data/tmax/train.jsonl --harbor-tasks-dir /data/tmax/tasks \
  --task-id task_004783_a0ba89ed --task-id task_008572_85bf1cba
```

## Launch

Use the ARM64 Miles image on GB300. Install the supported Harbor fork with uv
(the fork has a workspace dependency):

```bash
uv pip install 'harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0'
```

Configure the internal sandbox service as described in the
[sandbox-service skill](https://github.com/radixark/rdxa_skills/blob/main/plugins/rdxa-skills/skills/sandbox-service/SKILL.md).
A GPU devbox needs tailnet connectivity. Verify the CPU golden round trip:

```bash
python scripts/sandbox_smoke/run.py --connector harbor --backend e2b
```

The recipe's E2B environment serializes cold builds of the same template to
avoid sibling HTTP 409s ([AgentENV #74](https://github.com/radixark/AgentENV/issues/74)).
Sandbox creation stays concurrent after the image is ready. This guard is
process-local and relies on this recipe's single rollout worker; it does not
coordinate builds across independent training jobs.

Commands use TMax's process-group timeout (`TERM`, then `KILL` after ten
seconds), with thirty seconds of extra E2B response-stream time to collect
the exit code and partial output. A normal command timeout returns code 124
to the agent. This adapter does not add transport retries: an uncertain
command outcome fails the trial rather than replaying possible side effects
or assigning an infrastructure failure a zero reward.
Output is captured in temporary regular files so background services do not
keep E2B's output pipes open after the foreground shell exits. Background
services survive a successful tool call; a foreground timeout still stops
its process group.

Log in to W&B on the node and set `WANDB_API_KEY` to enable the usual launcher
tracking configuration. The launcher removes the key from its generated
command; W&B uses the node's existing credentials. Only the sandbox key file
path is included in Ray's recorded runtime configuration.

```bash
# Set these to the interface actually present inside the devbox.
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
python -m examples.experimental.tmax_score_centering.run_qwen3_5_9b \
  --model-dir /data/models --data-dir /data/tmax --output-dir /data/runs
```

The launcher prepares Terminal-Bench evaluation tasks, downloads and converts
the model if needed, then submits through Miles' standard `execute_train`
interface. It saves checkpoints, rollouts,
trajectories, and Harbor trial results beneath `output_dir/run_id`.

For the first full-data run on four GB300 GPUs:

```bash
python -m examples.experimental.tmax_score_centering.run_qwen3_5_9b \
  --model-dir /data/models --data-dir /data/tmax --output-dir /data/runs \
  --num-rollout 3651 --rollout-batch-size 4 --samples-per-prompt 8 \
  --max-concurrent-samples 24 --max-steps 64 \
  --max-seq-len 32768 --max-response-len 16384 \
  --eval-interval 100 --eval-max-steps 64 --save-interval 25 \
  --extra-args '--rollout-shuffle --max-weight-staleness 4 --async-unused-samples-handler retry'
```

This consumes 32 accepted trajectories per update. The 3,651-update budget is
approximately one pass over 14,601 prompts; retries and rejected groups mean
it is not a guarantee that each prompt contributes exactly once. The initial
24-episode concurrency is a measurement point, not an established optimum.
Track completed trajectories per second, trainer wait time, weight staleness,
sandbox latency, and GPU memory before increasing it. GPU telemetry includes
the baseline evaluation and initialization, which must be excluded when
measuring steady-state training utilization.

## Terminal-Bench 2.0 evaluation

The launcher prepares `data_dir/eval/terminal-bench-2.0.jsonl` through Harbor's
registry and copies the pinned task directories into `data_dir/tasks`, using
the `terminal-bench-2.0__` prefix. The adjacent `.manifest.json` records the
source revisions. Evaluation tasks never enter `train.jsonl`.

| Setting | Default |
|---|---|
| Dataset | All 89 tasks from `terminal-bench@2.0` |
| Schedule | Before training, every 100 updates, and at the end |
| Attempts per task | 1; success rate estimates pass@1 |
| Temperature / top-p / sampling top-k | 1 / 1 / unrestricted |
| Context / per-turn generation cap | 32768 / 16384 tokens |
| Maximum agent turns | 64 |
| Engine allocation | Share the existing three rollout GPUs |

`--eval-interval`, `--samples-per-eval-prompt`, and `--eval-max-steps` are
launcher options. Shared-engine evaluation pauses new training-rollout
submissions and blocks training updates until evaluation finishes, keeping
weights fixed for each evaluation. Existing in-flight episodes may finish.
The generate semaphore caps training plus evaluation episodes to 12 with the
default topology. This avoids starting all 89 sandboxes at once.

Evaluation uses the vendored Vanillux evaluation prompts and the single bash
agent without subagents or compaction. It preserves task-defined working
directories, resources, agent timeouts, and verifier timeouts. The verifier's
final-state score is used even when the agent reaches its turn/token limit
without a submit marker, matching the reference evaluation behavior; training
still requires the submit marker. Some tasks allow hours, so a full evaluation
can take substantially longer than the training smoke. Infrastructure errors
remain failures rather than silently disappearing from the denominator.

To prepare evaluation data separately:

```bash
python -m examples.experimental.tmax_score_centering.prepare_eval_data \
  --output /data/tmax/eval/terminal-bench-2.0.jsonl \
  --harbor-tasks-dir /data/tmax/tasks
```

For an existing checkout, also pass `--task-dir` and `--registry-entry` (the
`terminal-bench@2.0` DatasetSpec JSON from Harbor's registry). Task instructions
alone enter the user prompt; verifier and oracle solution contents are not
included in the JSONL. Resource declarations and task files are copied intact.

## Data retention for every future run

Keep both `--save-debug-rollout-data` and `--save-debug-trajectory-data` enabled.
These are permanent defaults in this launcher, including when evaluation is
added. Miles' evaluation paths use the same saver and add an `eval_` prefix,
so training and evaluation at the same step do not overwrite each other:

| Data | Path below `output_dir/run_id` |
|---|---|
| Training samples, rewards, token IDs, masks, sampled and top-128 logprobs | `rollouts/<step>.pt` |
| Evaluation samples and rewards | `rollouts/eval_<step>.pt` |
| Training conversations | `trajectories/<step>.jsonl` |
| Evaluation conversations | `trajectories/eval_<step>.jsonl` |
| Token columns for the Miles dashboard | `dashboard_columns/rollout_<step>.parquet`, `rollout_eval_<step>.parquet` |
| Environment, agent, and verifier logs | `trials/<trial_id>/` |

Evaluation files are created when an evaluation actually runs. Before releasing a devbox, archive
these directories to persistent storage or download them locally. Devbox
`/scratch` alone is not durable storage. Keep the run configuration and task
IDs alongside the archive so results can be matched to their checkpoint.

## Smoke settings and limits

| Setting | Smoke default |
|---|---|
| Trainer / rollout GPUs | 1 / 3, TP=1 |
| Updates | 3 |
| Prompts per update / attempts per prompt | 2 / 4 |
| Concurrent episodes | 12, independent of update batch size |
| Per-request generation cap / total sequence cap | 16384 / 32768 tokens |
| Tool turns | 16 |
| Temperature / top-p / top-k | 1 / 1 / unrestricted |
| Learning rate / gradient clipping | 1e-6 constant / 1 |
| Loss / candidate count | score_centering / 128 |
| Advantages | GRPO mean centering, no standard-deviation normalization |
| KL / entropy / importance sampling | 0 / 0 / none |

The launcher uses
`max_seq_len` for both the SGLang context limit and the trainer token budget;
`max_response_len` caps each model turn, including its reasoning and tool call.

The published TMax shape uses 32 samples for each of eight prompts, much
longer trajectories, and DPPO. This example uses GRPO plus score centering.
It does not establish published-quality
reproduction, FP32 LM-head parity, large-concurrency throughput, or policy
improvement from a few smoke updates. Restore the published budgets and tune
concurrency only after validating the full data and gradient path.

The full Terminal-Bench baseline has not completed. Sandbox command-connect
failures and state-consistency observations were recorded in
[AgentENV #75](https://github.com/radixark/AgentENV/issues/75). The issue was
reported closed, but the service fix has not been validated with this recipe.
Generic infrastructure retry and circuit-breaker handling remain follow-up
work; a failed trial can still terminate the run.
