---
title: Advanced Features
sidebarTitle: Overview
description: Systems-level features for large-scale and long-running RL.
---
This section covers the Miles features that the Core-features section of the
homepage points at: low-precision training (FP8 / MXFP8 / NVFP4 / INT4 QAT),
Rollout Routing Replay for MoE, fast weight updates over P2P RDMA,
disaggregated RL rollout through an external service, fault tolerance,
speculative decoding, and LoRA training and serving.

<CardGroup cols={2}>

  <Card title="Low Precision RL" icon="bolt" href="/advanced/low-precision">

    Unified block-wise FP8, MXFP8, and NVFP4 recipes with matched training and
    rollout precision.

  </Card>

  <Card title="INT4 QAT" icon="microchip" href="/advanced/int4-qat">

    W4A16 quantization-aware training for fitting large models on a single
    8-GPU node.

  </Card>

  <Card title="Rollout Routing Replay (R3)" icon="network-wired" href="/advanced/miles-router">

    Capture expert routing during inference and replay during training. The
    mechanism that keeps MoE RL stable.

  </Card>

  <Card title="Speculative Decoding" icon="rocket" href="/advanced/speculative-decoding">

    Draft + target speculative rollout, with online MTP-SFT for the draft.

  </Card>

  <Card title="On-Policy Distillation" icon="graduation-cap" href="/advanced/on-policy-distillation">

    Train a student on its own rollouts while matching teacher token
    probabilities through SGLang or Megatron teacher modes.

  </Card>

  <Card title="Disaggregated RL Rollout" icon="server" href="/advanced/disaggregated-rollout">

    Scale rollout across clusters and regions through an independent service,
    with versioned policy publication and request attribution.

  </Card>

  <Card title="LoRA Training and Serving" icon="sliders" href="/advanced/lora">

    Train LoRA adapters with SFT or RL and serve them through SGLang from the
    same checkpoint.

  </Card>

</CardGroup>
