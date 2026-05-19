---
title: Contributing
description: PR conventions, code layout, and how reviews work.
---

# Contributing

Miles is open source under the LICENSE file in the repo. We accept community
contributions of every size — bug reports, doc fixes, new model recipes, full features.

## Repository layout

```text
miles/
├── train.py                  # synchronous entry point
├── train_async.py            # fully-async entry point
├── miles/                    # the package itself
│   ├── backends/             # Megatron, SGLang, training (loss / GRPO / PPO / ...), experimental FSDP
│   ├── ray/                  # Ray actors + rollout driver
│   ├── rollout/              # rollout / data source / filters
│   ├── router/               # Miles Router (FastAPI proxy)
│   └── utils/                # async, types, IO, distributed helpers, arguments.py
├── miles_plugins/            # opt-in plugins
│   ├── mbridge/              # mbridge integration
│   ├── megatron_bridge/      # megatron-bridge integration
│   └── models/               # model wrappers (GLM4, GLM5, Qwen3.5, Qwen3-Next, ...)
├── examples/                 # the recipes documented in this site
├── scripts/
│   ├── models/               # per-model MODEL_ARGS bash files
│   ├── amd/                  # AMD-specific launchers
│   ├── tools/                # script-side utilities (e.g. verify_chat_template.py)
│   ├── run-*.sh              # bash launch scripts
│   └── run_*.py              # Python launch scripts
├── tools/                    # ckpt converters, calibrators, debug tools
├── tests/                    # pytest suite (fast / ci / e2e / utils)
├── docker/                   # Dockerfiles
└── docs/                     # the source of this site
```

## Local dev loop

```bash
# Inside the radixark/miles container
cd /root/miles
git remote add me git@github.com:<your_user>/miles.git
git checkout -b feat/awesome

# Edit code
pip install -e . --no-deps           # editable install picks up changes
pytest tests/test_my_thing.py -xvs    # run the relevant tests

git add -p && git commit -m "feat: short imperative description"
git push me feat/awesome
gh pr create --title "..." --body "..."
```

## Style

* **Python:** [`ruff`](https://docs.astral.sh/ruff/) for linting, `black` for
  formatting (line length 100). `pre-commit` is wired up.
* **Type hints:** use them. We're strict about new code.
* **Docstrings:** Google style for public functions; one-liners for internal.
* **Imports:** `isort` ordering, no relative imports across packages.

## PR checklist

Before you click "Ready for review":

- [ ] `pre-commit run --all-files` passes.
- [ ] You added or updated tests for new behaviour.
- [ ] You ran `pytest -x` and it's green.
- [ ] If you touched the launch flags, `python3 train.py --help` still parses.
- [ ] If you added a public flag, it appears in [Server Arguments](../user-guide/cli-reference.md).
- [ ] If you added a new example, you wrote a real walkthrough (use
  [examples/index](../examples/index.md) as the structural template).

## Commit messages

We follow the conventional-commits style:

```
feat(rollout): add partial-rollout buffer
fix(megatron): correct fp32 marker on Qwen3.5 A_log
docs: clarify FP8 rationale for MoE
test: cover R3 routing replay
```

The first line is < 70 chars. Body explains *why*, not *what* — the diff already
shows what.

## Issue triage

We label issues with:

| Label | Meaning |
|---|---|
| `good first issue` | Self-contained, no system knowledge needed |
| `help wanted` | We'd love community PRs |
| `bug` | Reproducible breakage |
| `enhancement` | Feature request |
| `discussion` | Design conversation, not yet a task |
| `needs-repro` | We can't reproduce — please provide a minimal example |

If you're new, sort by `good first issue`. Comment to claim before you start so we
don't double-up.

## Where to ask questions

* **Quick questions:** Miles channel of the [SGLang Slack](https://slack.sglang.ai).
* **Design discussions:** open a GitHub Discussion or an Issue with `discussion` label.
* **Security:** email security@radixark.ai (do not open a public issue).
