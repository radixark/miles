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

The installed `tbench2_env` must be `>=` the #1012 merge (04d259ea6), same as
step 2b below; the adapter drops every episode (with a warning) from a server
that doesn't carry that contract.

### 2b. Alternative: per-episode cloud sandboxes — Daytona, E2B, or AgentENV

Instead of one shared env server, every episode can get its **own cloud
sandbox**, built from the task's official image plus an env-server layer and
deleted when the episode ends. This keeps docker mode's per-task image
fidelity while leaving no resident infrastructure behind: no Docker socket,
no shared server to size or babysit, no cross-episode state. It costs a
sandbox creation per episode, plus one image or template build the first time
each task runs, which the provider caches from then on.

Whichever provider you pick, install `tbench2_env` **editable**: the recipe
bakes the installed source into each task image, so that install must carry
the same `>=` #1012 server contract as step 2. The launcher preflights the
installed source and fails fast on an older one.

```bash
git clone https://github.com/huggingface/OpenEnv.git   # >= the #1012 merge (04d259ea6, the full canonical contract for both modes); pin that sha if you need frozen reward semantics across a long run
pip install -e OpenEnv/envs/tbench2_env
```

Then skip step 2 and set two things: `OPENENV_TB2_TASKS_DIR`, the checkout to
build task images from, and `OPENENV_SANDBOX_BACKEND`, the provider to build
them on. Neither has a default — the provider decides whose quota a run
spends and which credentials have to be present — so setting one without the
other fails at launch.

Both providers authenticate the same way: the key in the environment
(`DAYTONA_API_KEY`, `E2B_API_KEY`), or else a file whose *path* the launcher
forwards. It never forwards the value, which ray's `runtime_env` records in
plaintext; the agent-function docstrings cover what that means on a
multi-host cluster.

**Daytona** builds each image declaratively per episode. A warm create takes
about a minute, and the first episode of each task spends about ten minutes
building its image, cached by definition hash after that.

```bash
pip install daytona
mkdir -p ~/.config/daytona && echo dtn_... > ~/.config/daytona/api_key   # or export DAYTONA_API_KEY
export OPENENV_TB2_TASKS_DIR=/workspace/terminal-bench-2   # the checkout from step 1
OPENENV_SANDBOX_BACKEND=daytona python run-openenv-tbench2.py
```

**E2B-compatible** providers (`e2b`, or `agentenv` as an alias for the same
leg) build one named template per task, and every later episode warm-starts
from it in seconds. The endpoint defaults to E2B Cloud; to drive a
self-hosted [AgentENV](https://github.com/kvcache-ai/AgentENV) deployment
instead, follow the [AgentENV recipe](../agentenv/README.md).

```bash
pip install e2b
export E2B_API_KEY=e2b_...     # or E2B_API_KEY_FILE (default ~/.config/e2b/api_key)
export OPENENV_TB2_TASKS_DIR=/workspace/terminal-bench-2
OPENENV_SANDBOX_BACKEND=e2b python run-openenv-tbench2.py
```

Because that template is a named artifact rather than a build cache, it can be
built ahead of the run. Doing so is optional, since the first episode of a
task builds it inline, but that inline build occupies a create-concurrency
slot, so an unbaked first rollout spends most of its wall clock building
images: `python tb2_sandbox_e2b.py --tasks-dir /workspace/terminal-bench-2 --all`.

Two sanity checks run without a GPU, and both honor
`OPENENV_SANDBOX_BACKEND`. [`scan_golden.py`](scan_golden.py) replays each
task's official solution through the full sandbox and scoring path; expect
82/89 to pass, since the rest have upstream-broken solutions, and pass
`--logs` to capture the failure evidence.
[`eval_tbench2_via_api.py`](eval_tbench2_via_api.py) runs the same agentic
loop with any OpenAI-compatible API standing in for the policy.

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
| `OPENENV_TB2_TASKS_DIR` | off | Switches on per-episode sandbox mode (section 2b) and overrides `--openenv-env-url`. Point it at the terminal-bench-2 checkout from step 1 |
| `DAYTONA_API_KEY` / `DAYTONA_API_KEY_FILE` | — / `~/.config/daytona/api_key` | Daytona key supply: the env value wins, otherwise the key file is read |
| `OPENENV_SANDBOX_BACKEND` | — | Required whenever `OPENENV_TB2_TASKS_DIR` is set: `daytona`, `e2b`, or `agentenv` (alias for `e2b`; section 2b). Only the selected leg's `OPENENV_*` settings below are read — the others are ignored silently |
| `E2B_API_KEY` / `E2B_API_KEY_FILE` | — / `~/.config/e2b/api_key` | E2B key supply, same file-path-forwarding contract as Daytona's |
| `E2B_API_URL`, `E2B_SANDBOX_URL` | E2B Cloud | Endpoint overrides — point both at a self-hosted AgentENV gateway |
| `OPENENV_DAYTONA_CREATE_CONCURRENCY` | `4` | Max in-flight creates on the daytona leg. Raise it against Daytona's creation rate limit, which the leg also retries with backoff |
| `OPENENV_E2B_CREATE_CONCURRENCY` | `4` | Max in-flight creates on the e2b leg. Size it to what the endpoint can host — one self-hosted AgentENV machine took 16 — and use `OPENENV_E2B_THROTTLE_PATTERNS` to name additional provider errors that should count as retryable capacity limits |
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
