---
title: User Guide
description: Concepts, launch script walkthrough, customization hooks, and a complete CLI reference.
---

# User Guide

| Page | What it covers |
|---|---|
| [Core Concepts](concepts.md) | The four objects in the training loop and the four-knob invariant. |
| [Training Backends](usage.md) | Choosing between Megatron-LM and FSDP, and each backend's plumbing. |
| [Training Script Walkthrough](training-script-walkthrough.md) | The eight `XXX_ARGS` arrays in a launch script, plus the execution modes (sync/async, colocation, dynamic sampling, partial rollout, BF16+FP8). |
| [Monitoring & Logging](monitoring.md) | wandb, structured logs, per-source breakdowns, profiling, router metrics. |
| [Customization](customization.md) | The 22 `--*-path` plug-points for custom Python — rollout, reward, filters, loss, hooks. |
| [Rollout Endpoints](rollout-endpoints.md) | The `/generate` endpoint and the OpenAI chat endpoint for agentic sessions. |
| [Agentic Chat Templates](agentic-chat-template.md) | Verifying and fixing the chat template so multi-turn rollout stays append-only. |
| [CLI Reference](cli-reference.md) | Every flag Miles accepts, grouped by subsystem. |

## Which pages do I actually need?

- **Training my first job** — read [Core Concepts](concepts.md), then [Training Script Walkthrough](training-script-walkthrough.md).
- **Tuning a running job** — [Training Script Walkthrough](training-script-walkthrough.md) in depth + [CLI Reference](cli-reference.md).
- **Plugging in a custom reward / rollout / filter** — skim [Core Concepts](concepts.md) for vocabulary, then go to [Customization](customization.md).
- **Contributor onboarding** — read top to bottom.
