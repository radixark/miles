---
title: User Guide
sidebarTitle: Overview
description: Concepts, launch scripts, customization hooks, and a complete CLI reference.
---
| Page | What it covers |
|---|---|
| [Core Concepts](/user-guide/concepts) | The four objects in the training loop and the four-knob invariant. |
| [Launch Script](/user-guide/launch-script) | What `python scripts/run_*.py` does, how a launch script is structured, and how to override a recipe. |
| [Argument Groups](/user-guide/argument-groups) | Where model args, `perf_args`, `grpo_args`, and the other launch-script flag groups belong. |
| [Fully Async RL](/user-guide/fully-async) | Continuous generation decoupled from training: the schedule, the data buffer, async eval, and the metrics to watch. |
| [Training Backends](/user-guide/training-backend) | Megatron-LM and FSDP: what each one owns, how to choose, parallelism, checkpoints, and hooks. |
| [Monitoring & Logging](/user-guide/monitoring) | wandb, structured logs, per-source breakdowns, profiling, router metrics. |
| [Customization](/user-guide/customization) | The `--*-path` plug-points for custom Python — rollout, reward, filters, loss, hooks. |
| [Generate Endpoint](/user-guide/generate-endpoint) | Custom generate functions that own tokens and loss masks via the raw `/generate` endpoint. |
| [Agentic Rollout (TITO)](/user-guide/agentic-rollout) | Configure an OpenAI-compatible agent loop with TITO trajectory assembly. |
| [Agentic Environments](/user-guide/environments) | Supplying an environment: dataset + reward, your own env via the plug points, or an external ecosystem. |
| [CLI Reference](/user-guide/cli-reference) | Every flag Miles accepts, grouped by subsystem. |

## Which pages do I actually need?

- **Training my first job** — read [Core Concepts](/user-guide/concepts), then [Launch Script](/user-guide/launch-script).
- **Tuning a running job** — [Launch Script](/user-guide/launch-script) in depth + [CLI Reference](/user-guide/cli-reference).
- **Plugging in a custom reward / rollout / filter** — skim [Core Concepts](/user-guide/concepts) for vocabulary, then go to [Customization](/user-guide/customization).
- **Contributor onboarding** — read top to bottom.
