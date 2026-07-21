---
title: Verifiers
description: Train on Verifiers environments with Miles.
---

Miles can train on a Verifiers environment in place of a prompt dataset. The
integration requires Python 3.11 or newer and Verifiers 0.2.0. Verifiers 0.2.1
requires OpenAI 2.9 or newer, while SGLang 0.5.15 pins OpenAI 2.6.1.

## Install

Install the optional dependencies with Miles and install the Prime CLI:

```bash
pip install -e '.[verifiers]'
uv tool install prime
```

The recommended workspace keeps local environment packages under `./environments`:

```text
workspace/
  environments/
    my-environment/
```

From the workspace root, install a local environment by name. For an environment from
the Environments Hub, authenticate and use its `user/environment` ID:

```bash
# Local: ./environments/my-environment
prime env install my-environment

# Environments Hub
prime login
prime env install user/my-environment
```

## Configure

Create a Verifiers `EnvConfig` TOML file. A minimal config selects a taskset:

```toml
[taskset]
id = "gsm8k-v1"
```

The config may also define the harness, runtime, judges, retries, and environment
limits supported by Verifiers. Verifiers applies per-rollout and group rewards before
the completed traces are returned to Miles.

The integration implements Verifiers' V1 environment contract. Legacy V0 environment
configs are rejected during startup.

## Run

Add one option to a normal Miles training command:

```bash
--verifiers-config /path/to/verifiers.toml
```

This uses the configured taskset instead of Miles prompt data. Environment behavior
comes from the Verifiers config, while Miles continues to own the model, sampling,
batching, concurrency, reward hooks, and optimizer settings. The Renderers library
formats environment messages with Miles' model and tokenizer settings.

The standard Miles rollout options keep their existing meaning:

| Miles option | Verifiers behavior |
|---|---|
| `--rollout-batch-size` | Number of task groups returned by each training rollout |
| `--n-samples-per-prompt` | Rollouts per training task |
| `--n-samples-per-eval-prompt` | Rollouts per evaluation task |
| `--rollout-shuffle` / `--rollout-seed` | Finite taskset order and sampling seeds |
| `--rollout-*` / `--eval-*` sampling options | Sampling and context limits |
| `--apply-chat-template-kwargs` | Typed template options passed to renderers |
| `--sglang-server-concurrency` | Physical engine capacity |
| Miles reward and filtering options | Applied after Verifiers scoring using the standard Miles hooks |

Evaluation covers every task in the taskset. Training cycles the taskset and advances
from the current Miles rollout ID when a run resumes.

## Environment Support

The adapter supports V1 environments that use the Chat Completions dialect with
text-only Renderers inputs. Tools require a model-specific renderer; use a registered
model identity in `--hf-checkpoint` or the existing `--sglang-tokenizer-path` option.
User simulators, multi-turn episodes, environment runtimes, per-rollout rewards, and
group rewards run through the standard Verifiers environment lifecycle.

Verifiers group rewards apply during both training and evaluation. Miles
`--group-rm` hooks remain training-only, matching the standard Miles rollout path.

## Limitations

`--partial-rollout` is not supported. A Verifiers episode owns live harness and
environment state and has no contract for resuming a partially executed episode. Miles
rejects this combination during argument validation.

`--chat-template-path` is also rejected because Renderers owns message formatting for
Verifiers environments. Use the checkpoint's native template and
`--apply-chat-template-kwargs` instead.

Streaming model requests, Responses and Anthropic dialects, multimodal inputs, OPD,
routing replay, and indexer replay are not supported by the transport. Miles
rejects the corresponding CLI options when they can be detected during startup.

Traces with multiple graph branches, including compaction, are rejected. Miles does
not currently preserve a trace's rollout-group boundary when it flattens multiple
training samples, which would make group-relative advantages incorrect.
