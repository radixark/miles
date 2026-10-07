# Review instructions

These instructions condense the repository rules in `.claude/rules/`. Apply each rule only to the paths it names, and when a finding comes from a rule, name the rule file in the finding so the author can read the full text.

## What Important means here

Important is for findings that break behavior or silently change a run:

- Correctness bugs, as in the default review guidance.
- A launch or request field that parses but never reaches its consumer, so a user setting becomes a silent no-op.
- A request path that writes to the parsed args or to a shared default dict instead of copying first.
- A request value that can change a launch constraint, or a merge that silently overrides, ignores, or forces a value without a comment at that spot saying which value wins and why.
- A model definition that cannot be loaded: a `.sh` definition, a `source scripts/models/<x>.sh` anywhere (including docker patches), a file name that differs from the `megatron_model_type` string, or import-time side effects.
- A launcher that silently loses snapshot coverage: not named `scripts/**/run_*.py`, or a public helper that is not `_`-prefixed (the snapshot suite treats every public non-`main` function as an entrypoint).

Everything in "Code style" and "Naming" below is Nit at most.

## Cap the nits

Report at most five Nits per review; if there are more, say "plus N similar items" in the summary. After the first review of a PR, post Important findings only, and Nits only on lines changed since the previous review.

## Do not report

- Anything pre-commit enforces: black, isort, ruff, autoflake, YAML checks, helm-lint, and the `ban-*` hooks (`mpu` getters, bare HF auto loaders, `huggingface-cli`).
- Generated files: `docs/examples/**` and the Examples tab of `docs/docs.json`, which `scripts/tools/sync_example_docs.py` mirrors from `examples/**/README.md` (review the README instead); `charts/*/Chart.lock`; recordings under `tests/snapshots/**` (read their diff as evidence, do not review their style).
- Vendored code: `miles/utils/chat_template_utils/templates/encoding_dsv32.py`.
- Style on lines the PR did not change, and code-style rules outside their paths (they do not cover `tests/`, `examples/` or `miles_plugins/`).
- Requests to port a legacy `examples/**/*.sh` launcher the PR merely edits.

## Always check

### Launch and request args (`.claude/rules/launch-and-request-args.md`)

Applies to `miles/**/arguments.py`, `miles/**/*args*.py`, `miles/utils/*.py`, `miles/backends/sglang_utils/**`, `miles/rollout/**`, `miles/router/**`, `miles/tinker/**`, `miles/ray/rollout/**`, `miles/ray/specs/**`, `examples/**/*.py`.

- Scope test: if two requests in the same run can correctly use different values, the field is a request field with an optional launch default; otherwise it is a launch constraint that request input must not change. Flag a new field placed on the wrong side.
- A new launch flag added because one call site wants a different value, or a new request field that lets a caller change a launch constraint, is a finding.
- Shared code must not choose a value by branching on a model or dataset name; that belongs in model-specific rules, the dataset config or per-engine-group config.
- A new field must extend the existing merge of defaults, launch args and request values, not add a parallel path or a second knob for the same value.
- Default precedence: the request wins for a request field; for a launch constraint a differing request value gets a clear error. Any departure needs a comment at that spot and a statement in the PR description.
- A violated launch constraint must fail at startup when it can be known then, not as a per-sample warning; when miles forces a value on an engine or router, it must check the response instead of silently padding or truncating.

### Launchers and model definitions (`.claude/rules/launch-and-model-scripts.md`)

Applies to `scripts/**`, `examples/**`, `miles/utils/external_utils/command_utils.py`, `miles/utils/external_utils/model_args_utils.py`.

- New launchers and model definitions are `.py`, never `.sh`.
- A model definition exposes one pure `model_args(**kwargs) -> str`; variants derive from the base with `load_sibling_model_args` and `moe_layer_freq` instead of copying it. Any environment knob it reads is listed in `CLEARED_ENV` in `tests/fast/launch_scripts/py_harness.py`.
- A launcher puts its knobs on a `ScriptArgs` dataclass extending `U.ExecuteTrainConfig`, reaches the shell only through `command_utils` (no hand-rolled `ray start` / `ray job submit`), and imports and runs its entrypoints with no GPU, checkpoint or network.
- Nothing machine-specific is hardcoded: no `/root/<Model>` or checkout path such as `/root/miles` (use `U.repo_base_dir` and the dir fields), no hardcoded wandb project or group (use `U.get_default_wandb_args`), no hand-rolled run id (use `U.create_run_id()`), no environment read at import time.
- `--num-gpus-per-node` is always passed, and `--rollout-num-gpus` is dropped wherever `--colocate` is set.
- A change to a launcher's argv or runtime env should come with the matching recording diff under `tests/snapshots/launch_scripts/py/`; a port from shell must name every intended difference in the PR description.
- The harness denylists in the launch-script tests do not grow without a test that states why the entry is there.
- Adding, renaming or deleting a launcher updates the docs that invoke it: `docs/models/**`, `docs/getting-started/quick-start.md`, `docs/platforms/**`.

### Code style (`.claude/rules/general-code-style.md`, Nit)

Applies to new or substantially modified code in `miles/**/*.py`, `scripts/**/*.py`, `tools/**/*.py`, `train.py`, `train_async.py`.

- Functions stay under roughly 100 lines and files under roughly 1,000; the main orchestration function reads like pseudocode.
- Prefer pure functions and immutable, read-only inputs; state lives in the lifecycle objects that own it. Stable derived values are computed once at the earliest valid lifecycle point.
- Pass the specific values a callee needs rather than a large object, unless that object is the established contract.
- Keyword arguments for ambiguous or boolean arguments; module-level, absolute imports unless a local import has a stated reason.
- Public, plugin and framework interfaces are not renamed without a migration plan.

### Naming (`.claude/rules/pool-cell-worker-names.md`, Nit)

Applies to `miles/**/*.py`, `tests/**/*.py`, `charts/**`.

- Deployment layers are run, pool (`pool_id`), cell (`cell_id`), pod, worker (`<cell_id>-<worker_in_cell_index>`). A spec declares a pool and is never an identity.
- New names for these layers must not use `fleet` or `group`; `LWS` appears only where an upstream literal is quoted. This does not cover unrelated established terms such as torch process groups or engine groups.

## Verification bar

- Behavior claims need a `file:line` citation in the source, not an inference from naming.
- "Field never reaches its consumer" needs the parse site plus evidence that nothing reads it.
- A launcher finding should point at the recording diff when one exists.

## Summary shape

Open the review body with a one-line tally such as `1 Important, 3 Nit`, and lead with "No blocking issues" when there are no Important findings.
