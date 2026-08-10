---
paths:
  - "scripts/**/*.py"
  - "scripts/**/*.sh"
  - "examples/**/*.py"
  - "examples/**/*.sh"
  - "miles/utils/external_utils/command_utils.py"
  - "miles/utils/external_utils/model_args_utils.py"
---

# Launch And Model Scripts

Conventions for launchers under `scripts/` and `examples/`, and for the model
definitions under `scripts/models/`. Follow them for a new launcher or model
definition, and when substantially modifying an existing one.
`.claude/rules/general-code-style.md` still applies to the python itself.

## Python, not shell

- **Write every launcher and every model definition as a `.py` file.** Do not
  add a `.sh` launcher, and do not add a `.sh` model definition. `scripts/`
  contains no shell launcher; the surviving `examples/**/*.sh` are legacy that a
  later pass will port, so extend them only when the alternative is worse, and
  never model a new script on them.
- **Model args are resolved by python file name.** `load_model_args` imports
  `scripts/models/<megatron_model_type>.py`; a shell definition cannot be found
  at all, and a `source scripts/models/x.sh` anywhere — including a docker patch
  — is a hard failure the hygiene tests reject.
- **Porting a shell recipe is a semantics-preserving move.** Record the old
  script, record the replacement, and compare the ray runtime env plus the
  `train.py` argv as a flag → values multiset. Every intended difference gets
  named in the change description; nothing else may differ.

## Model definitions (`scripts/models/`)

- **One file per model type, named exactly as the `megatron_model_type` string**
  the launcher passes to `execute_train` — dots and dashes in the name are fine
  (`glm4.5-355B-A32B.py`).
- **Expose a single `model_args(**kwargs) -> str`** returning the megatron flags
  as one string. Whitespace is collapsed by the loader, so lay the flags out
  over multiple lines with comments grouping them.
- **Keep it a pure function with no import-time side effects.** The file is
  imported by path, sometimes several times in one process.
- **Derive a variant from its base model instead of copying it.** Use
  `load_sibling_model_args` for layer-count or LoRA variants, and
  `moe_layer_freq(nlayers=..., first_k_dense_replace=...)` for
  `--moe-layer-freq`.
- **Any environment knob a model script reads must be listed in `CLEARED_ENV`**
  (`tests/fast/launch_scripts/py_harness.py`). The snapshots pin expanded model
  args, so an unfrozen knob lets a developer's exported override fail the suite.

## Launcher anatomy (`scripts/**/run_*.py`)

- **Name it `run_<family>_<size>[_<variant>].py`, snake_case.** Discovery is
  `scripts/**/run_*.py`; a launcher named anything else silently gets no
  snapshot coverage.
- **Open with a module docstring** covering what the script trains, what must
  already exist (converted checkpoint, external ray cluster, ...), an `Args:`
  block for the flags, and at least one runnable example invocation.
- **Put every knob on a `ScriptArgs` dataclass extending
  `U.ExecuteTrainConfig`.** Compute derived values in `__post_init__` or a
  `@property`, not at the call site.
- **Public module-level functions are entrypoints.** `prepare` downloads and
  converts, `execute` submits the job; the snapshot suite treats every public
  non-`main` function as an entrypoint and asserts it issues commands, so
  helpers must be `_`-prefixed.
- **Wire the CLI with `@U.dataclass_cli`** plus `typer.run(main)`, or a
  `typer.Typer()` with one `@app.command()` per role when the script has
  genuinely distinct modes (see the `train` / `worker` split in
  `scripts/run_nemotron_3_super_120b_a12b.py`).
- **Reach the shell only through `command_utils`.** `execute_train` owns the
  preamble, `ray start`, the runtime env and the job submission;
  `convert_checkpoint`, `hf_download_dataset`, `fp8_cast_bf16`,
  `start_mooncake_master` and `ssh_start_ray_workers` (passed as
  `before_ray_job_submit`) cover the rest. Do not hand-roll `ray start` /
  `ray job submit`; the self-executing launchers under `examples/` are legacy
  pinned by `tests/manual/launch_scripts/test_self_executing_launchers.py`.
- **Build the argv as grouped blocks** — checkpoint, rollout, perf, algorithm,
  optimizer, sglang, misc — then concatenate them. It keeps a recipe diffable
  against its shell ancestor and against its siblings.

## Nothing hardcoded about the machine

- **Directories are fields, not literals.** `--model-dir` (`/root/models`),
  `--data-dir` (`/root/datasets`) and `--output-dir` (`/root/shared_data`, from
  `ExecuteTrainConfig`). Never hardcode `/root/<Model>`, and never hardcode a
  checkout path such as `/root/miles` or `/workspace/miles` — use
  `U.repo_base_dir`, which a hygiene test enforces for shell scripts too.
- **wandb comes from `U.get_default_wandb_args`,** which engages only when
  `WANDB_API_KEY` is set. No hardcoded project or group names.
- **Run ids come from `U.create_run_id()`,** not a hand-rolled timestamp.
- **Read the environment at call time, not import time.** `num_nodes` uses a
  `default_factory` for exactly this reason; a value captured at import leaks a
  stale SLURM allocation into every snapshot.
- **Always pass `--num-gpus-per-node`,** so the GPU-count field is honoured
  under `--colocate` on a sub-8-GPU node. Drop `--rollout-num-gpus` wherever
  `--colocate` is set — `arguments.py` documents and implements it as ignored
  there.
- **A cluster that is already joined is expressed with
  `MILES_SCRIPT_EXTERNAL_RAY=1`,** not by deleting the `ray start` from the
  launcher.

## One launcher per recipe family

- **Merge near-duplicate recipes behind a table.** A frozen `_Recipe` dataclass
  keyed by a `Literal`-typed `--model-name` (or `--topology`, `--parallelism`)
  beats a copied file per variant; `scripts/run_qwen3_dense.py` covers six
  recipes this way.
- **Keep separate files when the recipes are different experiments.** Differing
  algorithm, parallelism, hardware assertion and tuning blocks mean merging
  would bloat the table and put a recipe in active use at risk — that is why
  `run_glm45_355b_a32b_8node.py` is its own file. If most fields of the merged
  table would be set by one variant only, do not merge.
- **Delete rather than half-port a recipe that provably never ran.** Say so in
  the change description instead of guessing at a configuration nobody has
  executed.

## Snapshot coverage is the review artifact

- **Every public entrypoint of every launcher has a recording** at
  `tests/snapshots/launch_scripts/py/<path>/<entrypoint>.txt`. Regenerate with
  `MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS=1 pytest tests/manual/launch_scripts`
  and read the resulting diff — it is what proves a launcher change does what it
  claims.
- **A launcher must import and run its entrypoints with no GPU, no checkpoint
  and no network.** Do the work inside the entrypoints, and express
  skip-if-already-done through `Path.exists`, which the harness freezes to the
  checkout and the sandbox.
- **Do not grow the harness denylists** (`_SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED`,
  `_SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH`, ...) without a test that
  states why the entry is there and fails once it stops being true.
- **Run `pytest tests/fast/launch_scripts tests/manual/launch_scripts`** for any
  change to a launcher, a model definition or `command_utils`.

## Documentation follows the launcher

- **Adding, renaming or deleting a launcher updates the pages that invoke it** —
  `docs/models/**`, `docs/getting-started/quick-start.md`, `docs/platforms/**`.
  Keep per-variant knob tables keyed by `--model-name`, and keep
  `hf download --local-dir` destinations in step with the directory defaults.
