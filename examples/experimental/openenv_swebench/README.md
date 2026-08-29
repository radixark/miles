# OpenEnv SWE-bench-style GRPO (GLM-4.7-Flash, single node)

Train GLM-4.7-Flash with GRPO on **SWE-bench-style** tasks (SWE-Rebench-V2
"donor" variants) through the HuggingFace [OpenEnv](https://github.com/huggingface/openenv)
env server. This is a sibling of [`../openenv`](../openenv) (Terminal-Bench-2):
same env server, same agentic loop, same reward-marker protocol — only the task
family and its grading differ. A miles-side adapter runs the multi-turn agentic
loop (`reset(task_id)` → { policy emits one shell command → `step(exec)` → feed
output back } → verify) against an **unmodified** OpenEnv env server; the reward
is the binary verifier result (1.0 if the task's `test_command` passes, else 0.0).

This guide targets a **single H200 node with 8 GPUs**. The run is colocated
(training + rollout on the same 8 GPUs): TP=4, EP=2, one SGLang engine per GPU.

## How this differs from the Terminal-Bench-2 example

The OpenEnv env server is generic: it **pulls** the image named in a task's
`task.toml [environment].docker_image`, copies the task dir into the container,
and exposes an `exec` step. Everything task-family-specific is in this adapter:

| | Terminal-Bench-2 (`../openenv`) | SWE-bench-style (here) |
| --- | --- | --- |
| Agent working dir | fixed `/app` | **auto-detected repo root** (`/app`, `/testbed`, `/<repo>`, …) |
| Verifier | `tests/test.sh` writes `/logs/verifier/reward.txt` | `tests/test.sh` prints `RESULT: PASSED`/`FAILED`, exits 0/1 (no `reward.txt`) |
| Reward source | value in `reward.txt` | derived from the `RESULT:` verdict line |
| No verdict recovered | drop sample | drop sample |

Both incompatibilities are handled in `swebench_agent_function.py` — the OpenEnv
repo is **not** patched or vendored.

## Prerequisites

- The node has Docker available (the env server launches one container per task).
- miles is installed and GLM-4.7-Flash weights are reachable (the launcher pulls
  `zai-org/GLM-4.7-Flash` from HF and converts it to `torch_dist` on first run).
- Install the OpenEnv env client (the same `tbench2_env` client the TB2 example
  uses; isolate it if its deps clash with the miles image):

  ```bash
  pip install -e <OpenEnv>/envs/tbench2_env
  ```

## 1. Make each task pullable (data prep)

The env server only ever **pulls** `task.toml [environment].docker_image`; it
does not build a Dockerfile. SWE-Rebench-V2 donor tasks ship an
`environment/Dockerfile` (e.g. `FROM docker.io/swerebenchv2/<repo>:<tag>`) and no
`docker_image`. So, once per task, build the image and record it — this is
offline data prep, **no code change**:

```bash
# For each task dir <task> in your pool:
docker build -t <registry>/<task>:latest <task>/environment
docker push  <registry>/<task>:latest
# then add to <task>/task.toml under [environment]:
#   docker_image = "<registry>/<task>:latest"
```

Many donor Dockerfiles only `FROM` a `swerebenchv2/...` base plus a `git reset`;
if the env host can pull that base directly you can point `docker_image` straight
at it, skipping the per-task build.

## 2. Build the prompt data

Emit one prompt row per `task_id` (the task dir name):

```bash
python make_swebench_data.py --tasks_dir /root/swebench_pool --output /root/swebench_train.jsonl
# add --n 8 for a small smoke subset
```

## 3. Start the env server

Run it in a separate shell (or off-node). Point `TB2_TASKS_DIR` at the pool from
step 1/2. Docker mode pulls the per-task images on first use:

```bash
# Raise the open-file limit first (see Notes): the WebSocket env server holds an
# FD per live session + Docker connection and leaks sockets on unclean
# disconnects, so the default 1024 soft limit is exhausted on a long run.
ulimit -n 1048576
TB2_MODE=docker TB2_TASKS_DIR=/root/swebench_pool MAX_CONCURRENT_ENVS=32 \
    python -m tbench2_env.server.app --port 8003
```

`MAX_CONCURRENT_ENVS` caps live sandboxes; keep it at or below the rollout batch
concurrency. Per-task containers are heavy on disk — if you'd rather not colocate
them with the GPU workload, run the env server on a separate Docker host and point
the launcher at it via `--openenv-env-url http://<env-host>:8003`.

## 4. Launch training

```bash
python run-openenv-swebench.py --openenv-env-url http://localhost:8003
```

Common overrides:

| Flag / env var | Default | Purpose |
| --- | --- | --- |
| `--openenv-env-url` | `http://localhost:8003` | Env server URL |
| `--prompt-data` | `/root/swebench_train.jsonl` | Prompt set from step 2 |
| `--num-rollout` | `40` | Number of GRPO steps |
| `OPENENV_MAX_TURNS` | `30` | Max agent turns per episode |
| `OPENENV_TASK_WORKDIR` | (unset → auto-detect) | Force a fixed repo path instead of detecting it |
| `OPENENV_SWEBENCH_TESTS_SRC` | `/task/tests` | Where the env stages the task's tests in the container |
| `OPENENV_EVAL_CMD` | (built-in) | Override the whole grading command (must print `__SB_REWARD__:<float>` last) |
| `OPENENV_MAX_ROLLOUT_TIME_SECONDS` | `3600` | Per-episode wall-clock cap; a straggler that exceeds it is terminated and the sample is dropped (the cap covers infra phases too, so a timeout can't produce a verdict — dropping avoids a false negative) |
| `OPENENV_SWEBENCH_TESTS_SNAPSHOT` | `/opt/.sb_pristine_tests` | Where the pristine grader tests are snapshotted at reset; grading reads this so mid-episode tampering with the staged tests can't reach the grader. Set `""` to disable |
| `OPENENV_SKIP_CLEANUP` | (unset) | Set `1` to skip the pre-run process reaper (use when another of your jobs is intentionally sharing the node) |
| `--dump-details <dir>` | off | Dump per-episode tokens/logprobs/masks/reward for inspection |
| `WANDB_KEY`, `--wandb-project`, `--wandb-team` | — | W&B logging |

## Notes

- **Reward signal.** The binary sparse reward needs a task subset where the base
  policy *sometimes* succeeds (advantage variance). If every task is always-fail
  or always-pass, GRPO sees a flat signal — curate a variance band (or use a
  stronger base) to see a learning climb.
- **Dropped samples.** If the verifier can't produce a `RESULT:` verdict (e.g. it
  can't locate the repo because the agent trashed the checkout), the episode has
  no recoverable reward and the sample is **dropped** rather than scored 0 — same
  policy as the TB2 adapter, so an infra/harness failure never becomes a false
  negative in training. A `OPENENV_MAX_ROLLOUT_TIME_SECONDS` timeout is treated
  the same way: the cap covers capacity/reset/generation/eval, so a timeout may
  fire before the verifier ran, which is a no-verdict case — dropped, not scored 0.
- **Verifier tampering.** The agent shares the task container with the grader
  assets, so it *could* edit them before grading. The adapter snapshots the
  pristine tests at reset and grades from that snapshot
  (`OPENENV_SWEBENCH_TESTS_SNAPSHOT`), which defeats naive tampering, but SWE-bench
  images run the agent as root so this is not a hard boundary — a fully robust
  setup would grade in a separate container the agent never touches.
- **`_step` vs. rollout.** W&B `_step` is an internal log-call index that advances
  several times per rollout; it is **not** the training step. Read the driver log's
  `rollout N:` counter for true progress.
- **Sandbox leakage.** Upstream OpenEnv creates task containers with `remove=False`
  and only tears them down on a clean session close, so an unclean disconnect
  (trainer crash) can orphan containers. Sweep stale containers between runs.
- **Open-file limit.** Unclean disconnects also leak socket FDs in the env server
  process. On a long run under the default 1024 soft limit the accept loop
  eventually fails every connection with `OSError: [Errno 24] Too many open
  files`. Start the server with a raised limit (`ulimit -n 1048576`, as in step 3).
