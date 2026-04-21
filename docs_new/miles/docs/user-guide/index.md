---
title: User Guide
description: A three-stage journey through Miles — understand the training loop, run a job, then extend it. Reference material sits at the bottom for lookup.
---

# User Guide

The User Guide is organised as a three-stage journey. Stages 1–2 are enough to train a model; stage 3 is where you drop in custom rollout, reward, or filter code without forking Miles. **Reference** is lookup, not reading.

| Stage | You are… | Pages |
|---|---|---|
| **1. Understand** | Building a mental model | [Core Concepts](concepts.md) · [Training Backends](usage.md) |
| **2. Run** | Launching and watching a job | [Training Script Walkthrough](training-script-walkthrough.md) · [Data & Datasets](data.md) · [Monitoring & Logging](monitoring.md) |
| **3. Extend** | Plugging custom code in | [Customization](customization.md) · [Rollout Endpoints](rollout-endpoints.md) · [Agentic Chat Templates](agentic-chat-template.md) |
| **Reference** | Looking a flag up | [CLI Reference](cli-reference.md) |

## 1. Understand

<div class="grid cards" markdown>

-   :material-compass-outline:{ .lg .middle } **[Core Concepts](concepts.md)**

    ---
    Rollout, actor, reference, reward. The training loop. The four-knob invariant. Start here.

-   :material-chip:{ .lg .middle } **[Training Backends](usage.md)**

    ---
    Megatron-LM vs FSDP. Checkpoint formats, parameter discovery, and each backend's customisation hooks.

</div>

## 2. Run

<div class="grid cards" markdown>

-   :material-script-text-outline:{ .lg .middle } **[Training Script Walkthrough](training-script-walkthrough.md)**

    ---
    Every argument group in a launch script, annotated. Plus sync/async, colocation, dynamic sampling, partial rollout, BF16+FP8.

-   :material-database:{ .lg .middle } **[Data & Datasets](data.md)**

    ---
    JSONL schema, metadata, multi-source datasets, and the filter hooks that run during rollout.

-   :material-chart-line:{ .lg .middle } **[Monitoring & Logging](monitoring.md)**

    ---
    wandb, structured logs, per-source breakdowns, profiling, router metrics.

</div>

## 3. Extend

<div class="grid cards" markdown>

-   :material-source-branch:{ .lg .middle } **[Customization (plug-points)](customization.md)**

    ---
    The twenty-plus `--*-path` plug-points where you can drop in custom Python without forking Miles.

-   :material-api:{ .lg .middle } **[Rollout Endpoints](rollout-endpoints.md)**

    ---
    The `/generate` endpoint for token-level control, and the OpenAI chat endpoint for agentic sessions.

-   :material-message-text:{ .lg .middle } **[Agentic Chat Templates](agentic-chat-template.md)**

    ---
    Verifying and fixing the chat template so multi-turn rollout stays append-only.

</div>

## Reference

<div class="grid cards" markdown>

-   :material-console:{ .lg .middle } **[CLI Reference](cli-reference.md)**

    ---
    Essentials up top, complete flag catalogue below. Use this when you know what you want and just need to look it up.

</div>

## Which pages do I actually need?

- **Training my first job** — read Stage 1 (two pages), skim Stage 2 Training Script Walkthrough, then go.
- **I have a job running and want to tune it** — Stage 2 in depth + Reference.
- **I want to plug in a custom reward / rollout / filter** — skim Stage 1 for vocabulary, jump straight to Stage 3.
- **Contributor onboarding** — read every stage top-to-bottom.
