---
title: Environments
description: How Miles trains on RL environments — datasets with rewards, self-wired environments, and optional external ecosystems.
---

Miles owns the training side of environment RL — batch orchestration, lossless
token-in/token-out recording, reward hooks, filtering — and is agnostic about
where the environment itself comes from:

- **No environment** — single-turn RLVR: a prompt dataset scored by the
  built-in rule-based rewards (math, ifbench, ...) or a custom reward function.
- **Your own environment** — plug your code into one of the three rollout
  layers described in [Integration shapes](#integration-shapes); most
  environments sit in the agent function, with the session server recording
  tokens (see [Rollout Endpoints](/user-guide/rollout-endpoints)).
- **An external ecosystem** — adopt a prebuilt connector from the table below;
  connectors occupy the same three layers.

| Integration | Plugs in at | Guide |
|---|---|---|
| [Harbor](https://github.com/harbor-framework/harbor) | agent function | [guide](/user-guide/harbor) |
| [NeMo-Gym](https://github.com/NVIDIA-NeMo/Gym) | agent function | [guide](/user-guide/nemo-gym) |
| [OpenEnv](https://github.com/huggingface/openenv) | agent function | [guide](/user-guide/openenv) |
| [Strands Agents](https://strandsagents.com/) | generate function | [example](https://github.com/radixark/miles/tree/main/examples/experimental/strands_sglang) |
| [Verifiers (Prime Intellect)](https://github.com/PrimeIntellect-ai/verifiers) | rollout function | [guide](/user-guide/verifiers) |
| [τ-bench](https://github.com/sierra-research/tau-bench) | generate function | [example](https://github.com/radixark/miles/tree/main/examples/experimental/tau-bench) |

Sandbox providers are a different axis: they provision the task containers
*inside* a connector rather than occupying a rollout layer.

| Sandbox provider | Used within | Guide |
|---|---|---|
| [AgentENV](https://github.com/kvcache-ai/AgentENV) | OpenEnv | [example](https://github.com/radixark/miles/tree/main/examples/experimental/agentenv) |
| [Daytona](https://www.daytona.io/) | OpenEnv, Harbor, NeMo-Gym | [example](https://github.com/radixark/miles/tree/main/examples/experimental/openenv) |
| [E2B](https://e2b.dev/) | OpenEnv | [example](https://github.com/radixark/miles/tree/main/examples/experimental/openenv) |
| [Modal](https://modal.com/) | OpenEnv | [example](https://github.com/radixark/miles/tree/main/examples/experimental/openenv) |

Everything above is experimental, and listed alphabetically.

## Integration shapes

The rollout stack is three nested plug-in layers (see
[Customization](/user-guide/customization)): each column in the table below
wraps the one to its left, so replacing an outer layer also takes over
everything an inner one would. A connector replaces exactly one layer.

✓ = the external framework takes it over; ○ = stays in Miles.

| | Agent function (innermost) | Generate function | Rollout function (outermost) |
|---|:---:|:---:|:---:|
| Plug-point flag | `--custom-agent-function-path` | `--custom-generate-function-path` | `--rollout-function-path` |
| Agent–environment loop | ✓ | ✓ | ✓ |
| Trajectory & token recording | ○ | ✓¹ | ✓¹ |
| Reward pathway (RM hooks, group rewards) | ○² | ○² | ✓ |
| Data source (prompts / taskset) | ○ | ○ | ✓ |
| Batch orchestration (grouping, filtering) | ○ | ○ | ✓ |
| Model, engines & weight updates, advantages, optimizer | ○ | ○ | ○ |

¹ Typically by speaking SGLang's native `/generate` (token IDs in and out)
rather than the session-server chat endpoint Miles' own recording uses.

² The environment may grade an episode itself (Harbor and τ-bench do); the
score still enters training through Miles' `Sample.reward` / RM hooks, and
group-level reward handling stays in Miles.
