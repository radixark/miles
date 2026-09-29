# Codex Instructions

Rules under `.claude/rules/` apply by path. Read and follow the ones matching the files you create or substantially modify:

- `miles/**/*.py`, `scripts/**/*.py`, `tools/**/*.py`, `train.py`, `train_async.py` → `.claude/rules/general-code-style.md` and `.claude/rules/no-getattr-defensive.md`.
- Any `*.py` or `*.cu` → `.claude/rules/comment-style.md`.
- `tests/**/*.py` → `.claude/rules/unit-test-admission.md`.
- A launcher under `scripts/` or `examples/`, or a model definition under `scripts/models/` → `.claude/rules/launch-and-model-scripts.md`.
- `miles/**/*.py`, `tests/**/*.py`, `charts/**` → `.claude/rules/pool-cell-worker-names.md`.

Before modifying a component listed in `.claude/rules/modify-component-must-read.md`, read the skill it names.
