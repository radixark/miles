---
title: Advanced Features
description: Systems-level features for large-scale and long-running RL.
---

# Advanced Features

This section covers the Miles features that the Core-features section of the
homepage points at: low-precision training (FP8 / MXFP8 / INT4 QAT), Rollout
Routing Replay for MoE, speculative decoding, and LoRA training and serving.

<CardGroup cols={2}>

  <Card title="Low Precision RL" icon="bolt" href="fp8-low-precision">

    The unified FP8 path: matched quantization between training and inference,
    BF16 backward and master weights.

  </Card>

  <Card title="INT4 QAT" icon="microchip" href="int4-qat">

    W4A16 quantization-aware training for fitting large models on a single
    8-GPU node.

  </Card>

  <Card title="Rollout Routing Replay (R3)" icon="network-wired" href="miles-router">

    Capture expert routing during inference and replay during training. The
    mechanism that keeps MoE RL stable.

  </Card>

  <Card title="Speculative Decoding" icon="rocket" href="speculative-decoding">

    Draft + target speculative rollout, with online MTP-SFT for the draft.

  </Card>

  <Card title="LoRA Training and Serving" icon="sliders" href="lora">

    Train LoRA adapters with SFT or RL and serve them through SGLang from the
    same checkpoint.

  </Card>

</CardGroup>
