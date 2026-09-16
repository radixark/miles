---
title: Contributing
description: Repository layout, the test suites, CI labels, and PR conventions for miles-diffusion.
---
miles-diffusion is the diffusion-model sibling of [miles](https://github.com/radixark/miles). It
shares miles' conventions — conventional commits, `pre-commit`, English-only PRs — but has its own
test topology, because almost everything meaningful needs GPUs.

## Repository layout

```text
miles_diffusion/
├── train_diffusion.py             # the entry point — the whole train loop, ~90 lines
├── miles/
│   ├── backends/
│   │   ├── fsdp_utils/            # the training half
│   │   │   ├── actor.py           # FSDPTrainRayActor: wrap, forward, loss, step
│   │   │   ├── configs/           # TrainPipelineConfig per model family
│   │   │   ├── models/            # native model packages (ltx) + diffusers parallel plans
│   │   │   ├── loss_hub/          # flow_grpo / nft / sft objectives
│   │   │   ├── sequence_parallel/ # USP = Ulysses × Ring
│   │   │   ├── monkey_patches/    # FSDP2 param-dtype patch (torch 2.11 gated)
│   │   │   ├── mixed_precision.py # per-parameter dtype map compilation
│   │   │   └── sde_step_backend.py# SDE dynamics + log-prob scoring
│   │   └── sglang_diffusion_utils/# the rollout half: engine wrapper + parity patches
│   ├── ray/                       # RolloutManager, train actor group, placement groups
│   │   └── data_conversion_hub/   # samples → train pairs
│   ├── rollout/                   # rollout fn, data source, rm_hub, filters, step strategies
│   ├── router/                    # miles router
│   ├── dashboard/                 # offline telemetry
│   └── utils/                     # arguments.py, types, metrics, debug tooling
├── scripts/                       # run_*.py launchers — Typer CLIs, also the e2e entry points
├── tests/                         # fast / fast-gpu / e2e / ci
├── docker/                        # Dockerfile + pinned version tags
└── docs/                          # this site
```

If you are adding a **model family**, you will touch:
`miles/backends/fsdp_utils/configs/<family>.py` (register with
`@register_train_pipeline_config`), optionally
`miles/backends/fsdp_utils/models/...` for a native package or FSDP plan, a launcher in
`scripts/`, and a page under `docs/models/`.

## Local dev loop

```bash
cd /root/miles_diffusion
git remote add me git@github.com:<your_user>/miles_diffusion.git
git checkout -b feat/awesome

pip install -e . --no-deps          # editable install picks up changes
pytest tests/fast -x -q             # CPU suite: no GPU, seconds
pytest tests/fast/backends/fsdp_utils/test_mixed_precision.py -xvs

git add -p && git commit -m "feat(fsdp): short imperative description"
git push me feat/awesome
gh pr create
```

## Tests

Four tiers, by what they need:

| Directory | Needs | What it is |
|---|---|---|
| `tests/fast/` | CPU only | Argument validation, config registry, dtype-map compilation, loss math, data splitting. The bulk of the suite. |
| `tests/fast-gpu/` | 1-2 GPUs | FSDP behaviour that only exists on device: param-dtype maps, hybrid shard, SP attention parity, LoRA weight sync. Includes ported upstream PyTorch FSDP tests. |
| `tests/e2e/short/` | 2-5 GPUs | Runs an actual launch script for a few rollouts and compares its **metric series** against a recorded standard. |
| `tests/ci/` | — | The harness itself: registry, label filter, suite runner, e2e standards. |

Run the CPU tier before every push; it is fast and catches most regressions. On a CPU-only box
without sglang's GPU kernels, install the stubs first the way CI does:
`uv pip install tests/ci/cpu_stubs`.

CI does not call `pytest` directly — it goes through the suite runner, which is also the way to
reproduce a CI failure locally:

```bash
python tests/ci/run_suite.py --hw cpu  --suite stage-a-cpu --labels run-ci-fsdp
python tests/ci/run_suite.py --hw cuda --suite stage-b-3-gpu-h200 --match-all-labels
```

Stage A (CPU) gates every GPU stage: a broken import should not burn GPU runner slots.

### Registering a test with CI

CI discovers tests by **AST-parsing** a `register_*_ci(...)` call at the top of the file — the
call is a runtime no-op and never executes.

```python
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])
```

| Parameter | Meaning |
|---|---|
| `est_time` | Seconds. Used to pack the suite. |
| `suite` | Which runner it lands on (below). |
| `labels` | Domain gate. Empty/omitted = runs on **every** PR. |
| `nightly` | Nightly only. |
| `disabled` | A string reason; the test is skipped. |

### Suites

| Suite | Runner |
|---|---|
| `stage-a-cpu`, `stage-b-cpu` | CPU |
| `stage-b-3-gpu-h200`, `stage-b-5-gpu-h200` | H200, split into 3-GPU and 5-GPU runners via per-runner `CUDA_VISIBLE_DEVICES` |
| `stage-c-3-gpu-h200`, `stage-c-5-gpu-h200` | Same hardware; stage C holds the long e2e runs |

### Labels

A test with a non-empty `labels` list runs only when the PR carries `run-ci-<label>` for one of
them. The canonical registry is `tests/ci/labels.py`:

| Label | Domain |
|---|---|
| `sglang-diffusion` | Engine wrapper and monkey-patch tests |
| `fsdp` | FSDP backend and config tests |
| `torch` | Ported PyTorch regression tests |
| `rollout` | Sampling, filters, step strategies |
| `ray` | Actors and placement groups |
| `router` | Routing decisions |
| `arguments` | Top-level argparse / `validate_args` |

Two meta-labels bypass the filter and run everything: `run-ci-all` and `run-ci-image`.

Adding a label means adding an entry to `tests/ci/labels.py` **and** creating the matching
`run-ci-<key>` label in GitHub repo settings. The workflow needs no edit — the stage job filters
at runtime.

### E2E metric standards

An e2e test declares the recipe to run, the arguments to run it with, and the metrics to check:

```python
register_e2e_ci(
    est_time=1200,
    suite="stage-c-3-gpu-h200",
    script="scripts/run_diffusion_grpo_sd3_ocr_sglang.py",
    args=["--num-rollout", "2"],
    metrics=["rollout/reward/raw_mean", "train/grad_norm", ...],
)
```

`script` is a Python recipe under `scripts/`, run with the same interpreter. Because the recipes
are Typer CLIs, CI configures them through `args` rather than through environment variables —
`env` still exists for anything the recipe reads from the environment instead.

This is also why launch scripts must stay runnable with no arguments and configurable through
their `ScriptArgs` dataclass: CI drives them the same way you do.

The recorded series live in `tests/ci/fixtures/e2e_standards/`. Because these runs use
`--deterministic-mode`, comparison is **strict, bit for bit** — which is exactly what makes them
useful, and also why any intentional numeric change requires re-recording the standard via the
`record-e2e-standards` workflow.

If your PR legitimately changes numerics, say so in the PR body and re-record. Do not loosen a
tolerance to make a test pass.

## Style

`pre-commit` is the enforcement point:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```

Hooks: `ruff` (with `--fix`), `autoflake`, `isort` (black profile), `black`, plus the standard
YAML / large-file / private-key checks.

- **Line length 119** (`black` and `isort` in `pyproject.toml`).
- **Python ≥ 3.12** — modern syntax is fine (`X | None`, `match`, PEP 695 where it reads well).
- **Type hints** on new code.
- **Comments explain why, not what.** The codebase leans on this heavily: most of the tricky code
  here exists because of a specific numeric mismatch, and the comment recording *which* mismatch
  is the most valuable line in the file.

## Docker changes

A PR touching `docker/Dockerfile` or `requirements.txt` triggers an image build; every GPU suite
then runs inside `radixark/miles_diffusion:pr-<num>` instead of `latest`, and a failed build stops
the matrix. The fresh build outranks a `ci-image-tag:` directive in the PR body. Fork PRs skip
the build and stay on `latest`. The tag is deleted when the PR closes.

## PR conventions

**English only** — title, body, commit messages, code comments.

Conventional commits, first line under ~70 characters:

```
feat(rollout): add epoch-global SDE step strategy
fix(fsdp): keep RoPE freq caches on CUDA for Qwen-Image
docs(models): document the LTX-2 velocity conversion
test(ci): cover the label filter
```

The body explains **why**; the diff already shows what.

### Before requesting review

- [ ] `pre-commit run --all-files` passes.
- [ ] `pytest tests/fast -x -q` is green.
- [ ] `python3 train_diffusion.py --help` still parses (any argparse change).
- [ ] A new public flag is documented in [CLI Reference](/diffusion/user-guide/cli-reference).
- [ ] A new model family has a page under `docs/models/`.
- [ ] Numeric changes are called out, and e2e standards re-recorded if they moved.
- [ ] New behaviour has a test, registered with the right suite and label.

## Where to ask

- Design discussion: open a GitHub Issue or Discussion on
  [radixark/miles_diffusion](https://github.com/radixark/miles_diffusion).
