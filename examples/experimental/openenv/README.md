# OpenEnv Terminal-Bench-2 GRPO (GLM-4.7-Flash, single node)

Train GLM-4.7-Flash with GRPO on the HuggingFace [OpenEnv](https://github.com/huggingface/openenv)
**Terminal-Bench-2 (tbench2)** environment. A miles-side adapter runs the multi-turn
agentic loop (`reset(task_id)` → { policy emits one shell command → `step(exec)` →
feed output back } → `evaluate`) against an unmodified OpenEnv env server; the reward
is the binary pytest result (1.0 = all tests pass, else 0.0).

This guide targets a **single H200 node with 8 GPUs**. The run is colocated
(training + rollout on the same 8 GPUs): TP=4, EP=2, one SGLang engine per GPU.

## Prerequisites

- Somewhere for the task environments to run, per step 2: an account with one of
  the hosted sandbox providers, a self-hosted AgentENV deployment, or a Docker
  host for the shared env server.
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

## 2. Provision the task environment

Every episode gets its **own cloud sandbox**, built from the task's official
image plus an env-server layer and deleted when the episode ends. That leaves
no resident infrastructure behind — no Docker socket, no shared server to size
or babysit, no cross-episode state — at the cost of one sandbox creation per
episode. (A single shared env server is still supported; see the last
subsection.)

Terminal-Bench-2 pins a different official image per **task**, which the recipe
honors rather than flattens. Two consequences follow on every provider: nothing
is shared across tasks, so all 89 pay their own first build; and since those
base images live on Docker Hub, which the providers' builders pull anonymously,
a first full-suite build can hit Docker Hub's anonymous pull limit — build a
few tasks at a time if it does.

Whichever provider you pick, install `tbench2_env` **editable**: the recipe
bakes the installed source into each task image, so that install must carry
the `>=` #1025 server contract. The launcher preflights the installed source
and fails fast on an older one.

```bash
git clone https://github.com/huggingface/OpenEnv.git   # >= the #1025 merge (38b2a3135: canonical contract for both modes, and a missing verdict is an error rather than reward 0); pin that sha if you need frozen reward semantics across a long run
pip install -e OpenEnv/envs/tbench2_env
```

Then set two things: `OPENENV_TB2_TASKS_DIR`, the checkout to build task images
from, and `OPENENV_SANDBOX_BACKEND`, the provider to build them on. Neither has
a default — the provider decides whose quota a run spends and which credentials
have to be present — so setting one without the other fails at launch.

Every provider authenticates the same way: the credential in the environment
(`DAYTONA_API_KEY`, `E2B_API_KEY`, or Modal's `MODAL_TOKEN_ID` +
`MODAL_TOKEN_SECRET` pair), or else a file whose *path* the launcher forwards.
It never forwards the value, which ray's `runtime_env` records in plaintext;
the agent-function docstrings cover what that means on a multi-host cluster.
A partially-supplied credential (one half of Modal's token pair) is treated as
missing rather than usable.

### Daytona

Builds each image declaratively per episode, straight from the image
definition: the first episode of a task pays the build and later ones hit
Daytona's build cache, keyed by definition hash. No named snapshot is involved,
so nothing accumulates against the org's snapshot quota — and correspondingly
there is nothing to pre-build ahead of a run.

```bash
pip install daytona
mkdir -p ~/.config/daytona && echo dtn_... > ~/.config/daytona/api_key   # or export DAYTONA_API_KEY
export OPENENV_TB2_TASKS_DIR=/workspace/terminal-bench-2   # the checkout from step 1
OPENENV_SANDBOX_BACKEND=daytona python run-openenv-tbench2.py
```

### E2B or AgentENV

Builds one **named template** per task, from which every later episode
warm-starts. The endpoint defaults to E2B Cloud; to drive a self-hosted
[AgentENV](https://github.com/kvcache-ai/AgentENV) deployment — the Firecracker
microVM platform whose native API *is* the E2B API — set `E2B_API_URL` and
`E2B_SANDBOX_URL` and follow the [AgentENV recipe](../agentenv/README.md).
`agentenv` is accepted as an alias for this backend.

```bash
pip install e2b
export E2B_API_KEY=e2b_...     # or E2B_API_KEY_FILE (default ~/.config/e2b/api_key)
export OPENENV_TB2_TASKS_DIR=/workspace/terminal-bench-2
OPENENV_SANDBOX_BACKEND=e2b python run-openenv-tbench2.py
```

Because the template is a named artifact rather than a build cache, it is the
one backend that can be built ahead of the run. Doing so is optional — the
first episode of a task builds it inline — but that inline build occupies a
create-concurrency slot, so an unbaked first rollout spends most of its wall
clock building images:

```bash
python tb2_sandbox_e2b.py --tasks-dir /workspace/terminal-bench-2 --all
```

### Modal

Expresses the recipe as one image layer per command, so editing the recipe
re-runs only the layers below the edit. There is no bake step: layer hashes
are themselves the cache key, so the first create for a task warms exactly
what later creates hit.

```bash
pip install modal
modal token new    # writes ~/.modal.toml; or export MODAL_TOKEN_ID + MODAL_TOKEN_SECRET
export OPENENV_TB2_TASKS_DIR=/workspace/terminal-bench-2
OPENENV_SANDBOX_BACKEND=modal python run-openenv-tbench2.py
```

Two things here are Modal's alone. Its sandbox timeout **cannot be extended**,
so `OPENENV_MODAL_SANDBOX_TTL_S` is a hard ceiling on one episode and must
exceed `OPENENV_MAX_ROLLOUT_TIME_SECONDS`; orphans are reclaimed by
`OPENENV_MODAL_IDLE_TIMEOUT_S` rather than by a keepalive thread (the
[materialization module](tb2_sandbox_modal.py) explains why that suffices, and
carries the snippet for sweeping a shared workspace). And for the Docker Hub
limit above, Modal takes registry credentials directly: hand
`Image.from_registry` a `modal.Secret`.

### Alternative: one shared env server

Rather than a sandbox per episode, one long-lived server can serve them all,
launching a container per task on its own Docker host. It is what the upstream
TB2 harness does, and it needs no sandbox platform at all — just Docker.
(Self-hosted AgentENV also avoids a vendor account, but is a platform you
deploy and operate.) The cost is resident infrastructure to size and clean up
after: see the shared-server entries under Notes.

```bash
# Raise the open-file limit first (see Notes): the WebSocket env server holds an
# FD per live session + Docker connection and leaks sockets on unclean
# disconnects, so the default 1024 soft limit is exhausted on a long run.
ulimit -n 1048576
TB2_MODE=docker TB2_TASKS_DIR=/workspace/terminal-bench-2 MAX_CONCURRENT_ENVS=32 \
    python -m tbench2_env.server.app --port 8003
```

Run it in a separate shell, and leave `OPENENV_TB2_TASKS_DIR` unset so the
launcher uses `--openenv-env-url` instead. `MAX_CONCURRENT_ENVS` caps live
containers; keep it at or below the rollout batch concurrency. Those containers
are heavy on disk, so if you'd rather not colocate them with the GPU workload,
run the server on a separate Docker host and point the launcher at it with
`--openenv-env-url http://<env-host>:8003`. The same `>=` #1025 `tbench2_env`
contract applies: the adapter drops every episode (with a warning) from a
server that doesn't carry it.

## 3. Sanity-check the environment (optional, no GPU)

Skippable — training runs without it — but a provisioning mistake skipped here
resurfaces as a rollout that burns GPUs to score zeros. Both scripts exercise
exactly what step 2 provisioned and honor the same `OPENENV_SANDBOX_BACKEND`,
so both cost provider time; scope them to a few tasks unless you want the full
sweep. [`scan_golden.py`](scan_golden.py) replays each task's official solution
through the full sandbox and scoring path; expect 82/89 to pass, since the rest
have upstream-broken solutions, and pass `--logs` to capture the failure
evidence. [`eval_tbench2_via_api.py`](eval_tbench2_via_api.py) runs the same
agentic loop with any OpenAI-compatible API standing in for the policy, which
also gives a baseline solve-rate to compare training against.

## 4. Launch training

```bash
export OPENENV_TB2_TASKS_DIR=/workspace/terminal-bench-2
OPENENV_SANDBOX_BACKEND=e2b python run-openenv-tbench2.py   # or daytona / modal
```

Against a shared env server instead, name its URL and leave
`OPENENV_TB2_TASKS_DIR` unset:

```bash
python run-openenv-tbench2.py --openenv-env-url http://localhost:8003
```

Common overrides. Every `OPENENV_*` knob below is read **once, when the rollout
worker imports the adapter** — a worker is a fresh process per run, so changing
one mid-run would change nothing. The exceptions are `OPENENV_TB2_TASKS_DIR`,
which the launcher injects through ray's `runtime_env` and each episode
re-reads, and `OPENENV_E2B_THROTTLE_PATTERNS`, consulted each time a create
fails.

| Flag / env var | Default | Purpose |
| --- | --- | --- |
| `--openenv-env-url` | `http://localhost:8003` | Shared env server URL; ignored in per-episode sandbox mode |
| `--prompt-data` | `/root/tbench2_train.jsonl` | Prompt set from step 1 |
| `--num-rollout` | (launcher) | Number of GRPO steps |
| `OPENENV_MAX_TURNS` | `30` | Max agent turns per episode |
| `OPENENV_MAX_ROLLOUT_TIME_SECONDS` | `3600` | Per-episode wall-clock cap; a straggler that exceeds it is terminated and scored 0 |
| `--dump-details <dir>` | off | Dump per-episode tokens/logprobs/masks/reward for inspection |
| `WANDB_KEY`, `--wandb-project`, `--wandb-team` | — | W&B logging |

Per-episode sandbox mode (step 2). The two switches come first, then each
provider's credentials, then the knobs — the ones shared by every backend
(`<BACKEND>` is `DAYTONA`, `E2B`, or `MODAL`), then each provider's own.

| Flag / env var | Default | Purpose |
| --- | --- | --- |
| `OPENENV_TB2_TASKS_DIR` | off | Switches on per-episode sandbox mode and overrides `--openenv-env-url`. Point it at the terminal-bench-2 checkout from step 1 |
| `OPENENV_SANDBOX_BACKEND` | — | Required whenever `OPENENV_TB2_TASKS_DIR` is set: `agentenv` (alias for `e2b`), `daytona`, `e2b`, or `modal`. Only the selected backend's settings below are read — the others are ignored silently |
| `DAYTONA_API_KEY` / `DAYTONA_API_KEY_FILE` | — / `~/.config/daytona/api_key` | Daytona key supply: the env value wins, otherwise the key file is read |
| `E2B_API_KEY` / `E2B_API_KEY_FILE` | — / `~/.config/e2b/api_key` | E2B key supply, on the same contract |
| `E2B_API_URL`, `E2B_SANDBOX_URL` | E2B Cloud | Endpoint overrides — point both at a self-hosted AgentENV gateway |
| `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` / `MODAL_CONFIG_PATH` | — / `~/.modal.toml` | Modal credential supply. The token PAIR must be complete; otherwise the config file's *path* is forwarded, like the other providers' key files |
| `MODAL_PROFILE`, `MODAL_ENVIRONMENT` | profile default | Which Modal profile / workspace environment the sandboxes are created in |
| `OPENENV_<BACKEND>_CREATE_CONCURRENCY` | `4` | Max in-flight creates. Size it to what the endpoint can host — one self-hosted AgentENV machine took 16; on Modal the ceiling above it is the plan's container concurrency |
| `OPENENV_<BACKEND>_CREATE_MAX_RETRIES` | `8` | How many throttled creates to retry before giving up (the backoff curve itself is not tunable) |
| `OPENENV_<BACKEND>_READY_TIMEOUT_S` | `300` | How long the env server has to answer /health after its sandbox exists |
| `OPENENV_E2B_SANDBOX_TTL_S` | `1800` | E2B sandbox TTL, re-armed by a keepalive thread while the creating process lives (Daytona's equivalent backstop is its own auto-stop/auto-delete, not a knob) |
| `OPENENV_E2B_THROTTLE_PATTERNS` | — | Extra comma-separated lowercase substrings that count as retryable capacity errors. Exists for self-hosted AgentENV, which words "at capacity" however its operator deployed it |
| `OPENENV_E2B_URL_SCHEME` | `https` | Scheme for the per-sandbox URL — set `http` for a plain-HTTP self-hosted AgentENV gateway |
| `OPENENV_MODAL_APP` | `openenv-tbench2` | App the sandboxes are created under; what a sweep scopes to in a shared workspace |
| `OPENENV_MODAL_BUILD_LOGS` | off | Stream Modal's image-build logs. Debug-only: it drives a process-wide output manager, so never set it with creates fanned out |
| `OPENENV_MODAL_CREATE_TIMEOUT_S` | `1800` | Build+create wall clock per Modal sandbox (the first create for a task builds its image, which Modal does not deadline itself) |
| `OPENENV_MODAL_SANDBOX_TTL_S` / `OPENENV_MODAL_IDLE_TIMEOUT_S` | `1800` / `300` | Modal lifetimes. The TTL is a HARD ceiling (Modal timeouts cannot be extended) and must exceed `OPENENV_MAX_ROLLOUT_TIME_SECONDS`; the idle timeout is what reclaims orphans, and an open tunnel connection counts as activity |

## Notes

- **Reward signal.** The binary sparse reward needs a task subset where the base
  policy *sometimes* succeeds (advantage variance). On the full TB2 suite,
  GLM-4.7-Flash's low base solve-rate yields a near-flat GRPO signal — use a
  variance-band subset (or a stronger base) to see a learning climb.
- **`_step` vs. rollout.** W&B `_step` is an internal log-call index that advances
  several times per rollout; it is **not** the training step. Read the driver log's
  `rollout N:` counter for true progress.
- **Shared server: container leakage.** Upstream OpenEnv creates task containers
  with `remove=False` and only tears them down on a clean session close (the idle
  reaper is off by default), so an unclean disconnect (trainer crash) can orphan
  containers. Sweep stale TB2 containers between runs, e.g. `docker rm -f` of any
  older than the episode wall-cap. Per-episode sandboxes have no equivalent: each
  provider arms its own TTL, so a dead caller's sandbox is reclaimed for you.
- **Shared server: open-file limit.** The same unclean disconnects also leak
  socket FDs in the env server process. On a long run under the default 1024 soft
  limit the accept loop eventually fails every connection with `OSError: [Errno
  24] Too many open files`, silently throttling rollouts. Start the server with a
  raised limit (`ulimit -n 1048576`, as in step 2); if a running server is already
  saturated, restart it with the higher limit.
