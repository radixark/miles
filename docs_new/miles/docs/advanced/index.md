---
title: Advanced Features
description: The systems work that turns Miles into a production framework.
---

# Advanced Features

If [Getting Started](../getting-started/installation.md) gets you running, this section is what
keeps you running at scale. These are the features that make weeks-long, trillion-parameter,
multi-rack RL feasible.

<div class="grid cards" markdown>

-   :material-network:{ .lg .middle } **[Miles Router (R3)](miles-router.md)**

    ---
    Capture expert routing during inference, replay during training. The fix for MoE RL collapse.

-   :material-flash:{ .lg .middle } **[FP8 & Low Precision](fp8-low-precision.md)**

    ---
    The unified FP8 path: training and inference at the same precision, no drift.

-   :material-chip:{ .lg .middle } **[INT4 QAT](int4-qat.md)**

    ---
    Fit a 1 TB-class model on a single H200 box. Quant-aware training keeps quality.

-   :material-rocket:{ .lg .middle } **[Speculative Decoding](speculative-decoding.md)**

    ---
    Draft + target speculative rollout, with online SFT for the draft model.

-   :material-shield-check:{ .lg .middle } **[Fault Tolerance](fault-tolerance.md)**

    ---
    Rank-level recovery, replay, and hot weight transfer.

-   :material-source-fork:{ .lg .middle } **[PD Disaggregation](pd-disaggregation.md)**

    ---
    Separate prefill and decode pools to maximise rollout utilisation.

-   :material-swap-horizontal:{ .lg .middle } **[P2P Weight Transfer](p2p-weight-transfer.md)**

    ---
    Sub-30-second weight sync from actor to rollout via NCCL P2P.

-   :material-puzzle-plus:{ .lg .middle } **[Backends Beyond Megatron](architecture-support.md)**

    ---
    Using FSDP, mcore, or your own backend behind Miles.

</div>
