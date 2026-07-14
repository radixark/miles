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
