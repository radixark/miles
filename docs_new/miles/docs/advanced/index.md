---
title: Advanced Features
description: Systems-level features for large-scale and long-running RL.
---

# Advanced Features

This section covers Miles features that come into play at larger scales or
longer runs: routing replay for MoE, low-precision training, weight-sync
mechanics, fault tolerance, and embedding HuggingFace modules in Megatron's
parallel pipeline.

<CardGroup cols={2}>
  <Card title="Rollout Routing Replay (R3)" icon="network-wired" href="miles-router">
    Capture expert routing during inference and replay during training. The
    mechanism that keeps MoE RL stable.
  </Card>

  <Card title="Low Precision RL" icon="bolt" href="fp8-low-precision">
    The unified FP8 path: matched quantization between training and inference,
    BF16 backward and master weights.
  </Card>

  <Card title="INT4 QAT" icon="microchip" href="int4-qat">
    W4A16 quantization-aware training for fitting large models on a single
    8-GPU node.
  </Card>

  <Card title="Speculative Decoding" icon="rocket" href="speculative-decoding">
    Draft + target speculative rollout, with online MTP-SFT for the draft.
  </Card>

  <Card title="Fault Tolerance" icon="shield-halved" href="fault-tolerance">
    Rollout-side health checks and engine recovery, gated by
    `--use-fault-tolerance`.
  </Card>

  <Card title="PD Disaggregation" icon="code-fork" href="pd-disaggregation">
    Separate prefill and decode pools for workloads where each phase has
    different bottlenecks.
  </Card>

  <Card title="P2P Weight Transfer" icon="arrows-left-right" href="p2p-weight-transfer">
    Direct rank-to-rank weight sync, an alternative to the broadcast path
    at large model scale.
  </Card>

  <Card title="Backends Beyond Megatron" icon="puzzle-piece" href="architecture-support">
    Wrap a HuggingFace implementation as a Megatron module without patching
    Megatron core.
  </Card>
</CardGroup>
