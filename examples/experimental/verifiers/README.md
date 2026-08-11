# Verifiers (Prime Intellect) rollout integration

Train on a [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) V1
environment instead of a Miles prompt dataset. Verifiers owns grouped episode
execution and reward computation; Miles keeps the model, sampling, engines and
weight updates, filtering, advantages, and the optimizer.

The adapter is a **rollout function** (`verifiers_rollout.py`): it replaces
Miles' batch-orchestration layer, runs `n` rollouts per task through Verifiers,
and returns ordinary Miles `Sample` groups. Renderers renders messages to token
ids and `MilesSGLangTransport` translates its wire format to Miles' SGLang
`/generate`, so training sees the exact sampled token ids and logprobs.

Requires Python 3.11+ and Verifiers 0.2.0. (Verifiers 0.2.1 requires OpenAI
2.9, while SGLang pins OpenAI 2.6.1.)

## Install

```bash
pip install -r examples/experimental/verifiers/requirements.txt
uv tool install prime
```

Install the environment itself with the Prime CLI, from a workspace that keeps
local environment packages under `./environments`:

```bash
# Local: ./environments/my-environment
prime env install my-environment

# Environments Hub
prime login
prime env install user/my-environment
```

## Configure

Write a Verifiers `EnvConfig` TOML. A minimal config selects a taskset:

```toml
[taskset]
id = "gsm8k-v1"
```

It may also define the harness, runtime, judges, retries, and environment
limits Verifiers supports. Legacy V0 configs are rejected at startup.

## Run

```bash
python examples/experimental/verifiers/run.py --verifiers-config /path/to/verifiers.toml
```

The launcher points `VERIFIERS_CONFIG` at the file and selects the adapter with
`--rollout-function-path verifiers_rollout.VerifiersRolloutFn` (the
`generate_rollout` function entry without
`MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`) plus
`--disable-rollout-global-dataset`. To wire a hand-rolled command, set the same
three things and put this directory on `PYTHONPATH`.

Standard Miles options keep their meaning:

| Miles option | Verifiers behavior |
|---|---|
| `--rollout-batch-size` | Task groups per training rollout |
| `--n-samples-per-prompt` | Rollouts per training task |
| `--n-samples-per-eval-prompt` | Rollouts per evaluation task |
| `--rollout-shuffle` / `--rollout-seed` | Taskset order and sampling seeds |
| `--rollout-*` / `--eval-*` sampling options | Sampling and context limits |
| `--apply-chat-template-kwargs` | Typed template options passed to renderers |
| `--sglang-server-concurrency` | Physical engine capacity |
| Reward and filtering options | Applied after Verifiers scoring |

`--hf-checkpoint` and `--sglang-tokenizer-path` provide the model and renderer
identity. Tools need a renderer registered for that identity, so pass a
registered model id in one of them when the checkpoint is a local snapshot.
Evaluation covers every task in the taskset; training cycles it and advances
from the current rollout id when a run resumes.

## Unsupported

The adapter raises at construction rather than producing wrong data:

- `--partial-rollout` — a live episode has no resume contract.
- `--chat-template-path` — renderers owns message formatting; use
  `--apply-chat-template-kwargs`.
- `--multimodal-keys`, `--use-opd`, routing replay, indexer replay — the
  transport does not carry that token metadata.
- Streaming, Responses/Anthropic dialects, and auxiliary relay routes.
- Traces with multiple graph branches, including compaction: Miles does not
  preserve a trace's rollout-group boundary when flattening, which would make
  group-relative advantages wrong.

## Evaluation

`run.py --eval-interval N` evaluates the whole taskset every N training
rollouts. Miles asserts that eval datasets are configured whenever
`--eval-interval` is set, so the launcher works around that by naming the
taskset and pointing the placeholder `--eval-prompt-data` at the EnvConfig it is
defined in; the adapter serves evaluation, so the built-in loader never opens
that path. Scoping the assertion Miles-side is worth revisiting once a second
rollout function owns its evaluation set.

Group rewards rank rollouts within a task, so evaluation needs
`--n-samples-per-eval-prompt >= 2` (the launcher's default).
