# AgenticRL-Bench with Kimi Code

Evaluate Kimi Code on AgenticRL-Bench tasks through a plain OpenAI-compatible
API and Harbor-managed sandboxes. This is a standalone eval: no training
process, session server, agent server, or interactive agent session is involved.
Harbor runs the original task and verifier; Kimi Code is the agent being tested.

## Prerequisites

- A checkout of AgenticRL-Bench or an adapted task set in Harbor format.
- Harbor with the environment you select installed. GPU sandboxes that allocate
  GPUs through E2B-compatible APIs need
  [harbor-framework/harbor#4398](https://github.com/harbor-framework/harbor/pull/4398).
- The sandbox provider's environment and credentials. For AgentENV, `aenv` is
  also required when a task environment is built from a Dockerfile.
- An OpenAI-compatible model endpoint and its API key in the local environment.
  The key is read from `KIMI_MODEL_API_KEY` by default and is never written to
  the command, result rows, or `run_config.json`.

## Run

Set the model endpoint and sandbox credentials locally, then select one or more
task directory names:

```bash
export POLICY_BASE_URL=https://api.example.com/v1
export KIMI_MODEL_API_KEY=...
export E2B_API_URL=https://sandbox.example.com
export E2B_API_KEY_FILE=~/.config/e2b/api_key

python examples/experimental/eval/agenticrl_bench_via_api/eval_agenticrl_via_api.py \
  --tasks-dir /path/to/agenticrl-bench/tasks \
  --task algorithm-partial_rollout-branch-reset-fidelity \
  --attempts 1 \
  --max-concurrent 1 \
  --model openai/kimi-k3.1
```

Use `--task` repeatedly to select several tasks. Without `--task`, the runner
selects every directory containing `task.toml`. Use `--environment` and
`--environment-kwarg KEY=VALUE` to configure the sandbox backend. For example,
GPU allocation uses `--environment-kwarg allocate_gpus=true`. Add `--system-dns`
when the Harbor client must use the operating system's DNS resolver. Artifact
uploads and downloads use streamed gzip transfers rather than buffering whole
files or batches in memory, avoiding the buffered-request timeout on large
checkpoints. For 16-hour open-ended tasks, set `--max-lifetime-secs 86400` so
sandbox lifetime covers the full agent budget and large-checkpoint collection.

The model endpoint must be reachable from the task environment. In particular,
a GPU sandbox with public-internet-only access cannot call a model endpoint
that exists only on a private network.

## Independent attempts

Run the first attempt alone and inspect its logs for infrastructure or task
problems before starting more attempts. A reward of zero is a valid model
result, not by itself an infrastructure failure.

```bash
# First attempt
python examples/experimental/eval/agenticrl_bench_via_api/eval_agenticrl_via_api.py \
  --tasks-dir /path/to/tasks --task TASK_NAME --attempts 1

# After the first attempt is healthy: attempts 2 and 3 only
python examples/experimental/eval/agenticrl_bench_via_api/eval_agenticrl_via_api.py \
  --tasks-dir /path/to/tasks --task TASK_NAME --attempt-start 2 --attempts 2
```

Each attempt gets its own Harbor job directory. Existing attempt directories
are never overwritten. `--max-concurrent` caps attempts in flight, including
across different tasks.

## Results

`--output-jsonl` receives one row per completed attempt with the task, attempt,
wall time, Harbor exit status, reward, process exit code, and artifact path.
The default is `agenticrl-kimi-code-results.jsonl`. `--jobs-dir` defaults to
`jobs/agenticrl-kimi-code` and holds each attempt's `run_config.json`,
`console.log`, Harbor result, agent log, and verifier output.

A zero reward does not make the runner fail. A nonzero runner exit indicates a
launch or execution error rather than an incorrect model solution.

## Offline tests

```bash
pytest tests/fast/examples/experimental/eval/agenticrl_bench_via_api -q
```
