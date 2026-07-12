# Multi-turn RL on CodeContests on AMD MI35x 

Self-contained example that RL-trains a Qwen3-30B-A3B model on the **CodeContests**
competitive-programming dataset. The model is the backend for the **mini-swe-agent**: each turn it
writes/edits `/app/solution.py` in a sandbox, runs it, inspects the output, and
iterates until it submits. Reward = whether the submitted program passes the hidden
tests. Training is **GRPO** on **token-in/token-out (TITO)** trajectories, so the
exact tokens the agent generated are the tokens trained on.

Default target: **Qwen3-30B-A3B** (MoE) in disaggregated **async** mode
(4 train + 4 rollout GPUs) on AMD-MI35x .

## How it works

Two containers share a Docker network (`swe-net`):

```
┌─ miles_swe (GPU) ──────────────────┐        ┌─ agent_env (Docker-in-Docker) ─┐
│  Megatron trainer (GRPO)           │        │  Harbor server (:11000)         │
│  SGLang rollout engine(s)          │        │   └ spawns 1 task container     │
│  Miles router (:31000)             │◀──────▶│     per trial (cc_base:v1)      │
│  TITO session server (:30000)      │  model │     mini-swe-agent runs there,  │
└────────────────────────────────────┘  calls │     grades via tests/test.sh    │
                                                └─────────────────────────────────┘
```

Per rollout step: Miles opens a TITO session → mini-swe-agent (inside a Harbor task
sandbox) calls back through the session server for each turn → Harbor grades the
final `/app/solution.py` → the reward + exact token trajectory become one training
sample → Megatron does a GRPO update → weights sync back to SGLang.

## Prerequisites

- A host with idle GPU(s) and a healthy ROCm stack, plus Docker.
- Two container images available to the host Docker daemon (override names via env):
  - trainer image (`MILES_IMAGE`: `rlsys/miles:MI350-355-latest`)
  - Harbor image (`AGENT_IMAGE`: `aigmkt/aai-2026-harbor:v1`)
- This repo checked out on the host (both containers bind-mount it identity-mapped).

## Run it

```bash
cd examples/experimental/qwen3-codecontests
WANDB_KEY=<optional> bash run_training_amd.sh
```

Everything is derived from the repo root and overridable via env. Useful flags:

```bash
bash run_training_amd.sh --skip-setup   # reuse running containers + installed miles
bash run_training_amd.sh --skip-data    # reuse prepared data + model
bash run_training_amd.sh --fresh-data   # wipe extracted tasks + jsonls, rebuild
bash run_training_amd.sh --reset        # restart the trainer between runs
bash run_training_amd.sh --teardown     # remove containers + task sandboxes
bash run_training_amd.sh -- --num-rollout 50 --save-interval 5   # passthrough to the launcher
```

Runtime data (tasks, trials, HF cache) lives under `$RUNTIME`
(default `<this dir>/runtime`); point it at a big disk if you have one:
`RUNTIME=/data/$USER/cc bash run_training_amd.sh`.

Training streams to the terminal and to `$WORK_DIR/train.log`; tail that file in a
second shell to watch progress.

## What the run does (in order)

1. **Host setup** — create `swe-net`, clean any stale containers.
2. **Containers** — launch `miles_swe` (trainer/SGLang/router/session server) and
   `agent_env` (Harbor), both bind-mounting this repo identity-mapped.
3. **Install miles** — `pip install -e . --no-deps` inside `miles_swe`.
4. **Harbor** — start `harbor/server.py`, wait for `/health`.
5. **Data prep** — `extract_codecontests.py` downloads `open-thoughts/CodeContests`
   into `$TASKS_DIR/<task_id>/`; `build_cc_jsonl.py` writes the per-difficulty JSONLs
   and the combined **`cc_train_all_sorted.jsonl`** used as `--prompt-data`.
6. **Fast spawn** — bake a tiny `cc_base:v1` (python3) once and rewrite each task
   Dockerfile to `FROM` it, collapsing per-trial container spawn toward the ~5s floor.
7. **Model prefetch** — pull the Qwen3 weights into the HF cache.
8. **Train** — async GRPO loop via `run-qwen3-codecontests.py`.

## Caveats

- The script **does not build** the trainer/Harbor images — it `docker run`s them; build/pull first.
- `cc_base:v1` is **local-only** (built on the host daemon Harbor uses); it must exist
  before task containers spawn (the data-prep step builds it).
- `--privileged --ipc host` is used for GPU access; ensure the host GPU stack is healthy.
- W&B is driven by `WANDB_KEY`/`WANDB_API_KEY` from the environment; empty ⇒ disabled.
  Never hard-code a key.

## Layout

| Path | What |
|---|---|
| `run_training_amd.sh` | **Entry point** — brings up both containers, installs miles, starts Harbor, preps data, and runs the async GRPO loop. ROCm/MI35x. |
| `run-qwen3-codecontests.py` | Training launcher (Typer); maps args → `launcher_args.py` → `execute_train`. |
| `launcher_args.py` | Dependency-free builder of the train-arg string (topology, GRPO, SGLang, session server). |
| `data_prep/extract_codecontests.py` | Download the HF dataset → per-task Harbor dirs; optional task-Dockerfile rewrite for fast spawn. |
| `data_prep/build_cc_jsonl.py` | Task dirs → per-difficulty JSONLs **and** the combined `cc_train_all_sorted.jsonl` (easy→hard). |
| `harbor/server.py` | FastAPI `/run` server wrapping Harbor's Trial API; maps results → reward/metrics. |
| `harbor/swe_agent_function.py` | Miles→Harbor bridge: dispatches a rollout to the Harbor server. |
| `harbor/generate.py` | Reward function, agent-metric aggregation, and the rollout class. |
| `harbor/codecontests.yaml` | mini-swe-agent config for the CodeContests contract (`/app/solution.py`, stdin→stdout). |
| `harbor/swe_net_override.yaml` | Compose override so task containers join `swe-net` and can reach the router. |

