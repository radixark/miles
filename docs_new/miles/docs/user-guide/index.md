---
title: User Guide
description: Concepts, launch script walkthrough, customization hooks, and a complete CLI reference.
---
| Page | What it covers |
|---|---|
| [Core Concepts](/miles/docs/user-guide/concepts) | The four objects in the training loop and the four-knob invariant. |
| [Argument Groups](/miles/docs/user-guide/argument-groups) | Where `MODEL_ARGS`, `PERF_ARGS`, `GRPO_ARGS`, and the other launch-script arrays belong. |
| [Training Backend](/miles/docs/user-guide/usage) | Megatron-LM as the training backend — parallelism, checkpoints, and hooks. |
| [Training Script Walkthrough](/miles/docs/user-guide/training-script-walkthrough) | The eight `XXX_ARGS` arrays in a launch script, plus the execution modes (sync/async, colocation, dynamic sampling, partial rollout, BF16+FP8). |
| [Monitoring & Logging](/miles/docs/user-guide/monitoring) | wandb, structured logs, per-source breakdowns, profiling, router metrics. |
| [Customization](/miles/docs/user-guide/customization) | The 22 `--*-path` plug-points for custom Python — rollout, reward, filters, loss, hooks. |
| [Rollout Endpoints](/miles/docs/user-guide/rollout-endpoints) | The `/generate` endpoint and the OpenAI chat endpoint for agentic sessions. |
| [Fully Async Rollout](/miles/docs/user-guide/fully-async) | Queue-backed rollout production, tuning knobs, and when to use `train_async.py`. |
| [Agentic Chat Templates](/miles/docs/user-guide/agentic-chat-template) | Verifying and fixing the chat template so multi-turn rollout stays append-only. |
| [CLI Reference](/miles/docs/user-guide/cli-reference) | Every flag Miles accepts, grouped by subsystem. |

## Which pages do I actually need?

- **Training my first job** — read [Core Concepts](/miles/docs/user-guide/concepts), then [Training Script Walkthrough](/miles/docs/user-guide/training-script-walkthrough).
- **Tuning a running job** — [Training Script Walkthrough](/miles/docs/user-guide/training-script-walkthrough) in depth + [CLI Reference](/miles/docs/user-guide/cli-reference).
- **Plugging in a custom reward / rollout / filter** — skim [Core Concepts](/miles/docs/user-guide/concepts) for vocabulary, then go to [Customization](/miles/docs/user-guide/customization).
- **Contributor onboarding** — read top to bottom.
