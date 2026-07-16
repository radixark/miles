# OpenEnv Terminal-Bench-2 GRPO (GLM-4.7-Flash, single node)

Train GLM-4.7-Flash with GRPO on the HuggingFace [OpenEnv](https://github.com/huggingface/openenv)
**Terminal-Bench-2 (tbench2)** environment. A miles-side adapter runs the multi-turn
agentic loop (`reset(task_id)` → { policy emits one shell command → `step(exec)` →
feed output back } → `evaluate`) against an unmodified OpenEnv env server; the reward
is the binary pytest result (1.0 = all tests pass, else 0.0).

This guide targets a **single H200 node with 8 GPUs**. The run is colocated
(training + rollout on the same 8 GPUs): TP=4, EP=2, one SGLang engine per GPU.

## Prerequisites

- The node has Docker available (the env server launches one container per task).
- miles is installed and GLM-4.7-Flash weights are reachable (the launcher pulls
  `zai-org/GLM-4.7-Flash` from HF and converts it to `torch_dist` on first run).
- Install the OpenEnv tbench2 env client (isolate it if its deps clash with the
  miles image):

  ```bash
  pip install -e <OpenEnv>/envs/tbench2_env
  ```

## 1. Build the prompt data

Clone the TB2 suite and emit one prompt row per `task_id`:

```bash
git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git /workspace/terminal-bench-2
python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 --output /root/tbench2_train.jsonl
# add --n 8 for a small smoke subset
```

## 2. Start the env server

Run it in a separate shell (or off-node — see note). Docker mode gives real TB2
fidelity; it needs the Docker socket and pulls the per-task images on first use:

```bash
# Raise the open-file limit first (see Notes): the WebSocket env server holds an
# FD per live session + Docker connection and leaks sockets on unclean
# disconnects, so the default 1024 soft limit is exhausted on a long run.
ulimit -n 1048576
TB2_MODE=docker TB2_TASKS_DIR=/workspace/terminal-bench-2 MAX_CONCURRENT_ENVS=32 \
    python -m tbench2_env.server.app --port 8003
```

`MAX_CONCURRENT_ENVS` caps live sandboxes; keep it at or below the rollout batch
concurrency. Per-task containers are heavy on disk — if you'd rather not colocate
them with the GPU workload, run the env server on a separate Docker host and point
the launcher at it via `--openenv-env-url http://<env-host>:8003`.

### 2b. Alternative: per-task Daytona cloud sandboxes (no Docker host)

Instead of one shared env server, the adapter can give **every episode its own
[Daytona](https://www.daytona.io/) cloud sandbox**, built from the task's official
image plus an env server layer and deleted when the episode ends. Same
per-task image fidelity as docker mode, with zero resident infrastructure (no
Docker socket, no shared server to size or babysit, no cross-episode state), at
the cost of per-episode sandbox creation (~1 min warm; the first episode of each
task builds its image in ~10 min, cached after that by definition hash).

The sandbox recipe lives upstream-side in `tbench2_env.task_snapshots` —
`tbench2_env` is OpenEnv's Terminal-Bench-2 environment package
(`envs/tbench2_env` in [huggingface/openenv](https://github.com/huggingface/openenv),
the same package step 2's shared server runs) — and needs its patched branch
(canonical `tests/test.sh` scoring built into `evaluate`, per-task WORKDIR
resolved server-side — the fidelity fixes proposed in
[openenv#965](https://github.com/huggingface/openenv/pull/965) /
[openenv#966](https://github.com/huggingface/openenv/pull/966)); the adapter then
scores via the standard `evaluate` action on this backend. Skip step 2 entirely
and set:

```bash
pip install daytona   # the SDK is imported lazily by tbench2_env, not installed with it
export DAYTONA_API_KEY=dtn_...
export OPENENV_TB2_TASKS_DIR=/workspace/terminal-bench-2   # the checkout from step 1
python run-openenv-tbench2.py
```

Infra sanity checks without touching a GPU (both live beside the launcher):
`scan_golden.py` replays each task's official solution through the full
sandbox + scoring path (`--logs` captures failure evidence; 82/89 of the TB2
suite pass, the rest have upstream-broken solutions), and
`eval_tbench2_via_api.py` runs the identical agentic loop with any
OpenAI-compatible API standing in for the policy.

## 3. Launch training

```bash
python run-openenv-tbench2.py --openenv-env-url http://localhost:8003
```

Common overrides:

| Flag / env var | Default | Purpose |
| --- | --- | --- |
| `--openenv-env-url` | `http://localhost:8003` | Env server URL |
| `--prompt-data` | `/root/tbench2_train.jsonl` | Prompt set from step 1 |
| `--num-rollout` | (launcher) | Number of GRPO steps |
| `OPENENV_MAX_TURNS` | `30` | Max agent turns per episode |
| `OPENENV_MAX_ROLLOUT_TIME_SECONDS` | `3600` | Per-episode wall-clock cap; a straggler that exceeds it is terminated and scored 0 |
| `OPENENV_TB2_TASKS_DIR` + `DAYTONA_API_KEY` | off | Per-task Daytona sandbox backend (section 2b); overrides `--openenv-env-url` |
| `OPENENV_DAYTONA_CREATE_CONCURRENCY` | `4` | Max in-flight sandbox creates (Daytona rate-limits creation) |
| `--dump-details <dir>` | off | Dump per-episode tokens/logprobs/masks/reward for inspection |
| `WANDB_KEY`, `--wandb-project`, `--wandb-team` | — | W&B logging |

## Notes

- **Reward signal.** The binary sparse reward needs a task subset where the base
  policy *sometimes* succeeds (advantage variance). On the full TB2 suite,
  GLM-4.7-Flash's low base solve-rate yields a near-flat GRPO signal — use a
  variance-band subset (or a stronger base) to see a learning climb.
- **`_step` vs. rollout.** W&B `_step` is an internal log-call index that advances
  several times per rollout; it is **not** the training step. Read the driver log's
  `rollout N:` counter for true progress.
- **Sandbox leakage.** Upstream OpenEnv creates task containers with `remove=False`
  and only tears them down on a clean session close (the idle reaper is off by
  default), so an unclean disconnect (trainer crash) can orphan containers. Sweep
  stale TB2 containers between runs, e.g. `docker rm -f` of any older than the
  episode wall-cap.
- **Open-file limit.** The same unclean disconnects also leak socket FDs in the
  env server process. On a long run under the default 1024 soft limit the accept
  loop eventually fails every connection with `OSError: [Errno 24] Too many open
  files`, silently throttling rollouts. Start the server with a raised limit
  (`ulimit -n 1048576`, as in step 2); if a running server is already saturated,
  restart it with the higher limit.
