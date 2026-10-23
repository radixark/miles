# Developing the TMax port

Start here when you pick up this port, whether you are a person or a coding agent. It says where everything is
documented, how the port hooks into Miles, how to change it safely, and how to validate a change before it reaches
a production run.

## Where things are documented

Each topic has one owner; the others link to it.

| Document | Owns |
|---|---|
| [`README.md`](README.md) | What the example does, how an episode runs, how to run it, results, and **Findings**: the gotchas found while porting |
| [`beaker/README.md`](beaker/README.md) | Running on Beaker: what a submission does, the `beaker:` config, run names and resume, images, sandbox pulls and mirrors, watching a run, failure triage |
| [`ROADMAP.md`](ROADMAP.md) | Open work by priority, decisions already taken, things not yet verified |
| [`INVESTIGATIONS.md`](INVESTIGATIONS.md) | Investigations from running the port: evidence, the options considered, open decisions |
| [`FAILURE_MODES.md`](FAILURE_MODES.md) | Index of every way a prod run has hung or failed, with its status (open, mitigated, fixed) and a link to its investigation |
| [`docs/deep-dive/open-instruct-comparison.md`](../../docs/deep-dive/open-instruct-comparison.md) | Every difference from open-instruct (framework, harness, sandboxes, Beaker, checkpoints, metrics), and the flag-by-flag mapping of the production recipe |
| [`docs/deep-dive/`](../../docs/deep-dive/index.md) | Miles internals: one RL step, every metric's formula, async and off-policy mechanics, agentic rollouts and the session server |
| This file | How the port is put together, how to develop and test it, conventions, history |

On the open-instruct side, branch `opd` holds the reference: the production script
`scripts/general_agent/terminal/rl/qwen35_9b_dppo_repro_4node_64k.sh` and the docs `docs/podman_sandboxes.md`,
`docs/registry_mirror.md`, `docs/tmax_rl_args_reference.md`, `docs/mason_reference.md` and
`docs/checkpoint_resumption.md`.

## Repository and branches

- **Remotes.** `origin` is the fork `github.com/shatu/miles`, branch `shashank/tmax-terminal-rl`; `upstream` is
  `github.com/radixark/miles`. The branch is rebased onto `upstream/main` from time to time; the last rebase, on
  2026-10-15, was onto `e30d9a8ec6`.
- **Prefer upstream fixes.** When Miles itself misbehaves, look for an upstream fix to rebase onto or cherry-pick
  before patching Miles core here. The port lives in `examples/tmax/`; its only core change is the head-node lookup
  in `miles/utils/ray_utils.py` (with `tests/fast/utils/test_ray_utils.py`), which `--pin-rollout-manager-to-head`
  needs on Beaker.
- **Rebase procedure:**
  1. Keep the old tip as `backup/tmax-terminal-rl-pre-rebase-<n>`.
  2. `git rebase upstream/main` and resolve the conflicts.
  3. Lint, and run the unit tests that can run locally.
  4. Run a Beaker smoke on the new base.
  5. `git push --force-with-lease origin shashank/tmax-terminal-rl`.

  Upstream changes often need a newer Miles image; check the flags the launcher passes against the image before a
  long run (see [Images](beaker/README.md#images)).

## How the port plugs into Miles

```
run_tmax_qwen3_5.py execute
  └─ train_async.py --fully-async        (train.py --colocate when synchronous)
       ├─ Megatron trainer actors        PPO on rollout log-probs, NCCL weight broadcast to the engines
       └─ rollout executor (on the head node)
            └─ agentic_tool_call.generate                 one call per sample
                 ├─ OpenAIEndpointTracer.create           a session on a session-server instance (v2)
                 ├─ call_agent_function(subproc)          a child process per episode
                 │    └─ tmax_agent_function.run          the episode loop
                 │         ├─ AsyncOpenAI → session server → SGLang (qwen3 / qwen3_coder parsers)
                 │         └─ TmaxSandbox → podman API (local unix socket or a worker's TCP forward)
                 └─ tracer.collect_samples                v2: the owning worker builds the training sample
            → generate.reward_func                        the reward the agent left in the metadata
            → zero-std filter, fully-async buffer (staleness), trainer
```

| Piece | Where | What to know |
|---|---|---|
| Launcher | `run_tmax_qwen3_5.py` | `ScriptArgs` holds every knob. A YAML config becomes `MILES_SCRIPT_*` env defaults, so command-line flags override it and an exported variable wins over the file. `prepare` downloads the dataset, extracts the task directories, writes the prompt JSONL (system prompt plus `metadata: {task_id, image}`), downloads the HF model and converts it to `torch_dist`. `execute` builds the train command in blocks (checkpoint, rollout, agent, optimizer, GRPO, W&B, perf, SGLang, misc), chooses `train.py` or `train_async.py`, and passes the sandbox settings to the agent processes as `TMAX_*` env vars |
| Generate function | Miles' `miles/rollout/generate_hub/agentic_tool_call.py` | Opens the session, runs the agent function, collects the session even if the agent failed, and marks the sample aborted when collection fails or no model call was recorded |
| Agent function | `tmax_agent_function.run` | The episode loop: [How an episode runs](README.md#how-an-episode-runs). Returns `{"reward", "agent_metrics"}`; session server v2 merges them into the sample's metadata |
| Sandbox | `tmax_sandbox.SandboxBase` and its backends | `SandboxBase` holds the episode logic both backends share: the persistent shell, observations, the withheld tests and the reward. `TmaxSandbox` (podman) uses docker-py against podman's Docker-compatible API; each episode picks a random non-benched daemon on a node below its sandbox cap, start and exec concurrency are per node, and a pull happens once per daemon and image. `tmax_gateway.TmaxGatewaySandbox` (gateway) puts the container on a LiteRegistry podman fleet through literegistry's own client (`literegistry-podman-client`), tracking the current literegistry release; commands are launched detached and waited for in bounded execs |
| Reward and metrics | `generate.py`, `tmax_metrics.py` | `reward_func` returns the agent's reward (`None` drops the group). `tmax_metrics`' filter hooks record each group before the zero-variance filter and its log hook (`--custom-rollout-log-function-path`) logs `rollout/{agent,tools,unfiltered,trained}/*` per batch, in both modes |
| Abort hook | `tmax_agent_function.abort` | Miles calls `<agent module>.abort(args)` when synchronous over-sampling has enough groups (`miles/utils/misc.call_agent_abort_hook`). It bumps a counter file and removes this rollout's containers; episodes race their work against the counter |
| Traces | `tmax_agent_function._write_trace` | One JSON line per episode in `<trace_dir>/<run>/traces-<host>.jsonl` (messages, reward, metrics), written under a cross-process lock |
| Training dumps | `--save-debug-rollout-data` / `--save-debug-train-data` | `dump_training_data: true` writes each step's trained batch next to the traces |
| Beaker | `beaker/submit.py`, `beaker/tmax_beaker_entry.sh`, `beaker/podman/` | [beaker/README.md](beaker/README.md) |

### The process model

The default `--custom-agent-function-mode subproc` gives every episode a process of its own. That shapes the code:

- **Module state lasts one episode.** `@cache`, module-level locks or counters do not span episodes. Limits shared
  across episodes (start and exec concurrency, one pull per image, sandboxes per node, the trace lock) are file
  locks from `miles.rollout.agentic.rollout_limits`, scoped to the rollout process by `current_scope()`. Benched
  daemons and the abort counter are files under the same scope.
- **Arguments and return values must pickle.**
- **Episodes are never cancelled.** `as_completed_async` would re-raise a cancellation, so an aborted episode stops
  by noticing the abort counter (`_race_abort`) and returns `reward=None`.
- **Edits to a live checkout reach running jobs.** Each episode re-imports the agent modules. A Beaker job therefore
  runs a pinned clone, never the weka checkout itself.

## Developing and testing

### What runs where

- **The local dev box cannot run Miles.** It is a Beaker session with 4× L40S and driver 570 (CUDA 12.8), and GPU
  containers cannot be nested inside a Beaker task. CUDA 13 itself is not the obstacle: CUDA 13 torch runs here (and
  on Jupiter) through NVIDIA's forward-compatibility driver, the image's `/usr/local/cuda/compat` on
  `LD_LIBRARY_PATH`. Training, SGLang and the session server only run in Beaker jobs.
- **Locally you can:**
  - lint;
  - run CPU unit tests;
  - sync and check the docs;
  - record the exact train command the launcher builds;
  - run podman sandboxes;
  - drive `tmax_agent_function.run` against a vLLM server, for agent-loop and sandbox changes and for baseline solve
    rates.
- **Local Python.** `/stage/.venv` is open-instruct's environment (torch, transformers, docker-py, vLLM 0.19; no
  SGLang, no Megatron). Run Miles' tests through `uv` on top of it, not by installing into it.

### Commands

```bash
# Lint (the versions pre-commit pins)
uvx --from black==24.3.0 black examples/tmax
uvx --from isort==5.13.2 isort examples/tmax
ruff check examples/tmax

# Unit tests that need no GPU (the gateway backend's need literegistry's client, and skip without it)
uv run --no-project --python /stage/.venv/bin/python --with "pytest-asyncio==0.24.0" \
    --with "literegistry-podman-client==0.1.7" python -m pytest -q --noconftest -p no:cacheprovider -o addopts="" \
    tests/fast/utils/test_ray_utils.py tests/fast/examples/tmax

# Tests that import SGLang: the wheel alone, no deps, in a scratch dir
uv pip install --no-deps --target /tmp/sgl_nodeps "sglang==0.5.21" --python /stage/.venv/bin/python
PYTHONPATH=/tmp/sgl_nodeps uv run --no-project --python /stage/.venv/bin/python --with "pytest-asyncio==0.24.0" \
    --with orjson --with IPython --with pylatexenc \
    python -m pytest -q --noconftest -p no:cacheprovider -o addopts="" tests/fast/examples/tmax

# After editing any examples/**/README.md: regenerate the docs site pages (pre-commit checks this)
python scripts/tools/sync_example_docs.py

# A podman API socket for local sandboxes
podman system service --time=0 unix:///run/podman/podman.sock &

# A local policy for agent-loop tests: small checkpoints, no reasoning parser for the tmax models
vllm serve allenai/tmax-2b --port 8011 --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- **Running episodes locally.** Call `call_agent_function(tmax_agent_function.run, mode="subproc",
  base_url="http://localhost:8011", prompt=..., request_kwargs={"temperature": 1.0, "max_tokens": 8192},
  metadata={"task_id", "image", "max_seq_len"})`. Build prompts as `prepare` does, and point `TMAX_TASK_DATA_DIR`
  at the extracted `task-data.tar.gz`. Without a Miles session server in between there is no token record, so this
  tests the loop and the sandbox, not the training sample.
- **Recording the train command.** The launcher is not in Miles' launch-script snapshot suite, which covers
  `scripts/**/run_*.py`. To see the command a config produces without GPUs, use the suite's harness in a throwaway
  test: `tests/fast/launch_scripts/py_harness` provides `freeze_environment`, `install_command_recorder`,
  `import_launch_script`, `call_entrypoint(module, "execute", <config fields>, tmp_path)` and `format_recording`.
  Run it with `--noconftest` and delete it afterwards.

### Validation ladder

Run each rung before the next, and keep jobs to 4 GPUs until the change is ready for a multi-node run.

1. **Local:** lint, unit tests, docs sync. Run the agent loop against vLLM on a few tasks if the harness changed.
2. **4-GPU smoke on Beaker** (`smoke_sync`, `smoke_async`, `smoke_opd`, `smoke_opd_async`): 3 steps. Check:
   - episode end reasons look normal and no sandbox failed to start;
   - `train/train_rollout_logprob_abs_diff` is about 0.007;
   - `grad_norm` is in range and weight syncs succeed.
3. **2-node smoke** (`smoke_2node`), with `-- --trace-dir <dir> --dump-training-data`. It exercises the replicas,
   the Ray join, the forwarded podman shards and the weight broadcast. Beaker may pack both replicas onto one
   machine; the worker IPs in `Ray cluster has … workers […]` tell you whether the sync crossed machines. Then
   check the trained samples against the traces with
   `HF_HUB_OFFLINE=1 python examples/tmax/tools/check_samples.py <run dir>/rollout_data/*.pt --trace-dir <run dir>`:
   - one sample per episode;
   - token, loss-mask and log-prob lengths consistent;
   - tool observations framed as in open-instruct;
   - trained stretches ending with `<|im_end|>`;
   - trained turns equal to the trace's `turns`, except for `context_full` episodes, which lose their last turn.
4. **Production** (`prod_4node`): submit, watch the first steps with `tools/watch_run.py <experiment> --until-steps 3`
   ([Watching a run](beaker/README.md#watching-a-run)), and push the branch once a few steps have trained.

### Conventions

- **A config is the experiment.** YAML fields of `ScriptArgs` plus a `beaker:` section. Commit it before
  submitting.
- **One owner per knob.** A setting lives in `ScriptArgs` (and the configs). `TMAX_*` env vars only carry it into
  the agent processes, and only the launcher writes them. Don't default a launcher field to a concrete value that
  would override what the entry script exports: an earlier default socket silently replaced the shard sockets, and
  every sandbox failed to start.
- **Harness constants say what they reproduce.** Keep the harness identical to open-instruct's. A deliberate
  difference goes in the [comparison](../../docs/deep-dive/open-instruct-comparison.md#harness-the-agent-loop), and
  the code comment names the open-instruct behavior it matches or departs from.
- **Never print a secret.** No `set -x` around tokens, no API keys on a logged command line. When checking a log,
  grep for the secret's value.
- **Docs follow the code.** Edit `examples/**/README.md` and run the sync script; never edit `docs/examples/` by
  hand. The deep-dive pages cite code at a commit; update the commit when you re-verify them.

## History

| Date | Milestone |
|---|---|
| 2026-09-24/25 | Port written; synchronous GRPO, fully-async GRPO and OPD (sync and async) validated on 4× B300 |
| 2026-10-13 | Rebased onto upstream; launcher configs; production recipe on one 8-GPU node |
| 2026-10-14 | Multi-node: Beaker replicas, sandboxes spread over nodes, first 4-node runs. A gradient reduce-scatter stalled at step 7 (10-min NCCL timeout) and a weight sync crawled at step 11 (SGLang's 5-min watchdog); a Docker Hub token in job logs was fixed (`set -x`); pull credentials and registry mirrors added; training-batch dumps added |
| 2026-10-15 | Gap audit against open-instruct. Timeouts raised and NCCL warnings turned on. Rebased onto upstream `e30d9a8ec6`: session server v2 with the namespaced radix cache, image `dev-202610151741`, 1 h synchronized replica start, per-submission run names. The old 4-node run reached step 118 before a weight-sync `ReadError` and a retry loop on unstarted replicas. 2-node v2 smoke validated: trained samples match v1's |
| 2026-10-16 | 4-node prod on the rebased code (`tmax_prod4node_qwen35_9b_20261016_001037`), with no session-collect failures. Branch force-pushed at `edf2c7ae3e`. Engine probe: `trtllm_mha` 1.75× faster than SGLang's default `triton`, and the fp32 LM head pays off only on both sides. Step 16 took 47 min while one node's engines crawled. The rollout executor segfaulted at step 24; Beaker resumed. Retry 1 reached step 34 before Ray declared a starved rollout node dead. Details in [INVESTIGATIONS.md](INVESTIGATIONS.md) |
