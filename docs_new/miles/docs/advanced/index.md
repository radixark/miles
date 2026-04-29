---
title: Advanced Features
description: Systems-level features for large-scale and long-running RL.
---

# Advanced Features

This section covers Miles features that come into play at larger scales or
longer runs: routing replay for MoE, low-precision training, weight-sync
mechanics, fault tolerance, and embedding HuggingFace modules in Megatron's
parallel pipeline.

<div class="grid cards" markdown>

-   :material-network:{ .lg .middle } **[MilesRouter and R3](miles-router.md)**

    ---
    Capture expert routing during inference and replay during training. The
    mechanism that keeps MoE RL stable.

-   :material-flash:{ .lg .middle } **[FP8 and Low Precision](fp8-low-precision.md)**

    ---
    The unified FP8 path: matched quantisation between training and inference,
    BF16 backward and master weights.

-   :material-chip:{ .lg .middle } **[INT4 QAT](int4-qat.md)**

    ---
    W4A16 quantization-aware training for fitting large models on a single
    8-GPU node.

-   :material-rocket:{ .lg .middle } **[Speculative Decoding](speculative-decoding.md)**

    ---
    Draft + target speculative rollout, with online MTP-SFT for the draft.

-   :material-shield-check:{ .lg .middle } **[Fault Tolerance](fault-tolerance.md)**

    ---
    Rollout health checks, training-side checkpoint recovery, and partial
    rollout reuse.

-   :material-source-fork:{ .lg .middle } **[PD Disaggregation](pd-disaggregation.md)**

    ---
    Separate prefill and decode pools for workloads where each phase has
    different bottlenecks.

-   :material-swap-horizontal:{ .lg .middle } **[P2P Weight Transfer](p2p-weight-transfer.md)**

    ---
    Direct rank-to-rank weight sync, an alternative to the broadcast path
    at large model scale.

-   :material-puzzle-plus:{ .lg .middle } **[Backends Beyond Megatron](architecture-support.md)**

    ---
    Wrap a HuggingFace implementation as a Megatron module without patching
    Megatron core.

</div>
