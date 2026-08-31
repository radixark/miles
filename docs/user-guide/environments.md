---
title: Agentic Environments
sidebarTitle: Overview
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
  tokens (see [Agentic Rollout (TITO)](/user-guide/agentic-rollout)).
- **An external ecosystem** — adopt a prebuilt connector from the table below,
  spanning coding, computer-use and tool-calling agents; connectors occupy the
  same three layers.

| Integration | Plugs in at | Guide |
|---|---|---|
| [Harbor](https://github.com/harbor-framework/harbor) | agent function | [guide](/user-guide/harbor) |
| [HUD](https://hud.ai) | generate function³ | [example](https://github.com/radixark/miles/tree/main/examples/experimental/hud) |
| [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | agent function | [guide](/user-guide/nemo-gym) |
| [OpenEnv](https://github.com/huggingface/openenv) | agent function | [guide](/user-guide/openenv) |
| [Strands Agents](https://strandsagents.com/) | generate function | [example](https://github.com/radixark/miles/tree/main/examples/experimental/strands_sglang) |
| [Verifiers (Prime Intellect)](https://github.com/PrimeIntellect-ai/verifiers) | rollout function | [guide](/user-guide/verifiers) |
| [τ-bench](https://github.com/sierra-research/tau-bench) | generate function | [example](https://github.com/radixark/miles/tree/main/examples/experimental/tau-bench) |

Sandbox providers are a different axis: they provision the task containers
*inside* a connector rather than occupying a rollout layer. Harbor accepts any
of its own environment backends when run
[in-process](https://github.com/radixark/miles/tree/main/examples/experimental/harbor)
(`HARBOR_ENV_TYPE` is passed through); the table lists the ones exercised.

| Sandbox provider | Used within | Guide |
|---|---|---|
| [AgentENV](https://github.com/kvcache-ai/AgentENV) | Harbor, OpenEnv | [example](https://github.com/radixark/miles/tree/main/examples/experimental/agentenv) |
| [Daytona](https://www.daytona.io/) | Harbor, HUD, NeMo Gym, OpenEnv | [example](https://github.com/radixark/miles/tree/main/examples/experimental/openenv) |
| [E2B](https://e2b.dev/) | Harbor, OpenEnv | [example](https://github.com/radixark/miles/tree/main/examples/experimental/harbor) |
| [Modal](https://modal.com/) | Harbor, OpenEnv | [example](https://github.com/radixark/miles/tree/main/examples/experimental/openenv) |

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

³ HUD's harness records per-turn token ids and sampling logprobs itself when
the inference server returns them, so the connector's job is stitching those
into one training sequence rather than recording. Computer-use observations
are screenshots, which Miles' session-server recording does not carry yet —
once it does, this connector can also sit in the agent function.
