---
title: User Guide
description: The mental model, the launch script, the data format, the customisation surface, and the reference.
---

# User Guide

The User Guide is organised as a reading order. Pages earlier in the list set up
vocabulary that later pages rely on. The last page — **CLI Reference** — is for
lookup, not reading.

<div class="grid cards" markdown>

-   :material-compass-outline:{ .lg .middle } **[Core Concepts](concepts.md)**

    ---
    Rollout, actor, reference, reward. The training loop. The four-knob invariant.
    Start here.

-   :material-chip:{ .lg .middle } **[Training Backends](usage.md)**

    ---
    Megatron-LM vs FSDP. Checkpoint formats, parameter discovery, and each backend's
    customisation hooks.

-   :material-script-text-outline:{ .lg .middle } **[Training Script Walkthrough](training-script-walkthrough.md)**

    ---
    Every argument group in a launch script, annotated. Plus sync/async, colocation,
    dynamic sampling, partial rollout, BF16+FP8.

-   :material-database:{ .lg .middle } **[Data & Datasets](data.md)**

    ---
    JSONL schema, metadata, multi-source datasets, and the filter hooks that run
    during rollout.

-   :material-source-branch:{ .lg .middle } **[Customization](customization.md)**

    ---
    The twenty-plus `--*-path` plug-points where you can drop in custom Python
    without forking Miles.

-   :material-api:{ .lg .middle } **[Rollout Endpoints](rollout-endpoints.md)**

    ---
    The `/generate` endpoint for token-level control, and the OpenAI chat endpoint
    for agentic sessions.

-   :material-message-text:{ .lg .middle } **[Agentic Chat Templates](agentic-chat-template.md)**

    ---
    Verifying and fixing the chat template so multi-turn rollout stays
    append-only.

-   :material-chart-line:{ .lg .middle } **[Monitoring & Logging](monitoring.md)**

    ---
    wandb, structured logs, per-source breakdowns, profiling, router metrics.

-   :material-console:{ .lg .middle } **[CLI Reference](cli-reference.md)**

    ---
    Essentials up top, complete flag catalogue below. Use this when you know what you
    want and just need to look it up.

</div>
