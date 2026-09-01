# Parallel checkpoint eval during SFT

This experiment connects pure SFT (`--debug-train-only`) to Miles' existing
asynchronous snapshot-eval path. Training uses the actor GPUs. A separate eval
fleet loads an HF export of the requested training step, and
`ParallelCommandEvalFn` starts the benchmark drivers whose per-command cadence
is due. Training continues while the eval runs.

The mental model is:

1. SFT finishes step 200.
2. Miles collectively exports `step_200` as an HF snapshot.
3. The eval engines load and verify weight version `200`.
4. Terminal Bench and HLE run concurrently against that pinned endpoint because
   both cadences are due at step 200.
5. Their JSON summaries are flattened into `eval/<command>/...` metrics at
   training step 200.
6. With overflow policy `skip`, a still-running eval causes the next scheduled
   point to be recorded as busy instead of slowing training.

## Training flags

Add these to an SFT launch using `train_async.py`:

```text
--debug-train-only
--rollout-num-gpus 0
--eval-num-gpus 8
--eval-num-gpus-per-engine 1
--eval-function-path examples.experimental.eval.parallel_sft.parallel_command_eval.ParallelCommandEvalFn
--eval-prompt-data external-command examples/experimental/eval/parallel_sft/eval_trigger.jsonl
--eval-interval 50
--eval-hf-dir /shared/eval-snapshots
--eval-max-in-flight 1
--eval-overflow-policy skip
--eval-keep-snapshots 1
--skip-eval-before-train
--eval-sglang-context-length 262144
--eval-sglang-served-model-name qwen-checkpoint
```

`--eval-prompt-data` satisfies Miles' generic eval CLI contract. The single-row
`eval_trigger.jsonl` is only a scheduling trigger; the command manifest owns the
actual benchmark inputs. Use a shared filesystem for
`--eval-hf-dir`, because the training actors export there and eval engines on
another node load from it.

Set the runner configuration before launch:

```text
MILES_PARALLEL_EVAL_CONFIG=/shared/config/parallel_eval.yaml
MILES_PARALLEL_EVAL_OUTPUT_DIR=/shared/eval-results
MILES_PARALLEL_EVAL_MODEL=qwen-checkpoint
MILES_PARALLEL_EVAL_AGENT_BASE_URL=http://public-checkpoint-router:30000/v1
```

`MILES_PARALLEL_EVAL_MODEL` and `--eval-sglang-served-model-name` must match.
You may omit both; the runner then uses the full `--hf-checkpoint` path, which
is SGLang's default served-model name.

`MILES_PARALLEL_EVAL_AGENT_BASE_URL` is the same checkpoint router exposed at an
address reachable from the Harbor agent-server host, such as a Tailnet ingress.
HLE runs directly from the `RolloutManager` and continues to use Miles' private
`{openai_base_url}`. Terminal Bench sends `{agent_openai_base_url}` to the remote
agent server. If the agent server is on the same network as the eval fleet, omit
the override and both placeholders resolve to the private router.

Copy `parallel_eval.example.yaml` and set the referenced Terminal Bench and HLE
environment variables. The manifest invokes HLE every 50 optimizer steps and
Terminal Bench 2.1 every 200 optimizer steps. `--eval-interval` must be the
greatest common divisor of the command cadences so Miles exports every needed
snapshot. An `interval_steps` cadence uses one-based optimizer steps: a value of
200 runs at Miles rollout IDs 199, 399, 599, and so on.

The bundled Terminal Bench driver evaluates all 89 tasks twice (178 trials) with
Terminus 2, temperature 1.0, top-p 0.95, top-k 20, 131,072 output tokens, and
262,144 context. Set `TB21_AGENT_SERVER_URL` to a Harbor-compatible agent server
whose task directory contains Terminal Bench 2.1. The bundled task list is
`terminal_bench_2_1_tasks.txt`; its 89-task count is validated before any
request is sent. The agent server owns sandbox setup, summarization, and its
three-hour agent timeout. The HLE command uses the same 131,072-token output
limit.

With `--eval-max-in-flight 1` and overflow policy `skip`, command cadences are
best effort: one eval fleet can serve only one checkpoint version at a time. If
the 178-trial Terminal Bench run is still using step 199 when step 249 is
reached, that newer checkpoint eval is skipped rather than changing weights
under the active agents or pausing training. Use a separately pinned eval fleet
if every HLE boundary must complete while a long Terminal Bench run is active.

## Driver contract

Commands are executed without a shell. The following placeholders are available
in every `argv`, `env`, and `metrics_path` value:

- `{checkpoint_dir}`
- `{agent_openai_base_url}` (external-agent-visible router override)
- `{litellm_model}` (`openai/` plus the configured model name)
- `{model}`
- `{openai_base_url}`
- `{output_dir}` (unique to the training step)
- `{rollout_id}`
- `{router_url}`
- `{training_step}` (one-based optimizer step)
- `{weight_version}`

Benchmark drivers run as subprocesses of the Ray `RolloutManager`. In a typical
two-node SFT job, that process is on the head/training node, not on the node that
hosts the eval SGLang engines. Any executable, local input, or credential source
used by a driver must therefore be readable in the `RolloutManager` process
namespace. The eval-engine nodes only serve the checkpoint endpoint.

The same values are exported as `MILES_EVAL_*` environment variables. A command
may write any JSON object to `metrics_path`. Numeric leaves are logged under
`eval/<command>/...`. A top-level `per_task` mapping is retained in the artifact
but omitted from W&B to avoid creating hundreds of metric keys. For per-sample
score reporting, emit this shape:

```json
{
  "metrics": {"accuracy": 0.42, "completed": 300},
  "rewards": [1.0, 0.0]
}
```

Stdout, stderr, and a copy of parsed metrics remain under the step output
directory. A failed driver is recorded with success 0 and its return code; it
does not terminate training or discard metrics from the other driver.

Those flattened numeric leaves are sent through Miles' normal tracking layer
with `eval/step` as the step key. When `--use-wandb` is enabled, they therefore
appear in the training run itself. The example publishes the primary benchmark
scores as `eval/tb21/accuracy` and `eval/hle_300_x4/accuracy`.

`terminal_bench_eval.py` counts errors and missing trials as incorrect for its
primary `accuracy`: `passes / (tasks * trials)`. Thus the configured full run is
`passes / 178`, rather than a more flattering pass rate over only successfully
graded trials. It also logs completed, graded, failures, errors, timeouts,
sequence-length exits, and problem-any-correct metrics. Per-task details and the
178 aligned rewards remain in the JSON summary artifact without creating 89
additional W&B time series.

`hle_eval.py` separates answer generation from grading. `--base_url` and
`--model` select the checkpoint endpoint. For a full HLE run, set
`--judge_base_url` to an independently hosted OpenAI-compatible endpoint
(including `/v1`) and set `--judge_model` to its served model name. SGLang's
OpenAI server and router are supported directly. The grader request uses a JSON
schema for `reasoning` and `correct`. The evaluated model is instructed to end
with `Final answer: ANSWER`; the driver extracts only that final line and never
sends the reasoning trace to the grader. Multiple-choice answers are scored
locally without consuming grader requests. Free-form grading requests contain
only the extracted candidate answer and the reference answer.

For models whose chat template supports it, pass `--disable_thinking` to render
the generation prompt with `enable_thinking=false`. This is useful for pipeline
smoke tests or checkpoints that fail to terminate their reasoning; normal HLE
evaluation leaves thinking enabled.

If the grader requires authentication, put its token in the environment variable
named by `--judge_api_key_env` (default `HLE_JUDGE_API_KEY`) so the secret is not
placed in command-line arguments. `--judge_max_qps` enforces a process-wide
minimum interval between request starts, including retries. For an endpoint with
a hard 2 QPS ceiling, use `--judge_max_qps 1.8` to leave scheduling margin.
Make sure the variable reaches the Ray job and its `RolloutManager`; setting it
only in the eval-server shell is insufficient. If a wrapper loads the token from
a mounted secret file instead, verify that file on the `RolloutManager` node
before launch. Never put the token in the YAML manifest, command arguments, or a
shared snapshot directory.

The example evaluates all 300 input rows four times and grades all 1,200
responses through the external endpoint:

```fish
set -x HLE_JUDGE_API_KEY <token-if-required>
python examples/experimental/eval/parallel_sft/hle_eval.py \
    --input /path/to/hle_text_only_300.jsonl \
    --base_url http://checkpoint-router:30000/v1 \
    --model qwen-checkpoint \
    --output_jsonl /path/to/hle_300.jsonl \
    --summary_json /path/to/hle_300_summary.json \
    --n_trials 4 \
    --concurrency 32 \
    --max_tokens 131072 \
    --temperature 1.0 \
    --judge_base_url http://external-grader:30000/v1 \
    --judge_model hle-grader \
    --judge_concurrency 32 \
    --judge_max_qps 1.8 \
    --judge_max_tokens 16384
```

Without `--judge_base_url`, the script retains its judge-free smoke behavior:
multiple-choice rows can be scored from an explicit final answer, while
free-form rows remain ungraded.
