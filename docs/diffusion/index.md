---
title: Miles-Diffusion Documentation
---
Miles-diffusion is [Miles](https://github.com/radixark/miles) for **diffusion models** — RL post-training for any
diffusion process, across image generation, video generation, and robotics.
[sglang-diffusion](https://github.com/sgl-project/sglang) serves the rollout, and the DiT trains under **FSDP2** on a
backend that co-evolves with Miles' own. Models load from a diffusers pipeline, or from a native package when a family
brings its own modeling. The shipped recipes are tuned and validated end to end. Custom rewards, losses, or rollout
functions plug in through a flag.

*"A journey of a thousand miles begins with a single rollout."* — For DiT models the rollout is a full denoising
trajectory, and miles-diffusion focuses on the system work that makes trajectory-level RL stable at scale, efficient,
and reproducible.

## Core features

- **Fast and stable support for the latest diffusion models.** Launch-ready recipes for Wan2.2-T2V-A14B, Qwen-Image,
  LTX-2.3, the Cosmos3 MoT omni family, and SD3.5. A per-family `TrainPipelineConfig` isolates model quirks so new
  architectures plug in without touching the trainer.
- **LoRA training with ipc-handle weight sync.** PEFT LoRA on the FSDP2 actor; each iteration ships only
  `lora_A`/`lora_B` pairs to the rollout engines over CUDA IPC and merges them engine-side — no full-weight transfer, no
  separate merge or conversion step. See [LoRA Training and Weight Sync](/diffusion/advanced/lora).
- **Quality control on three fronts.** Deterministic mode makes runs bitwise reproducible and backs the CI e2e
  regression suite; sglang-side monkey patches manage train/rollout alignment; and an FSDP2 param-dtype patch manages
  precision — by providing per-parameter level fp32 precision control over FSDP2 under the mixed-precision policy. See
  [Deterministic Training](/diffusion/advanced/deterministic) and [Dtype Control](/diffusion/advanced/dtype-control).
- **SFT, DiffusionNFT, and Flow-GRPO under one trainer.** The loss type, training-batch preparation, rollout function,
  and reward function are all **replaceable components**, so integrating a new algorithm — or swapping in your own
  customized component — is easy.
- **Sglang native.** Rollout runs **on the inference engine itself** — the sglang-diffusion serving stack — with RL
  support and optimizations living engine-side, and **train–inference consistency** is managed through a curated set of
  monkey patches that pin engine kernels to the numerics of the training-side diffusers forward to achieve maximized
  match.
- **Multiple parallelisms.** The rollout engines scale with **tensor and sequence parallelism** to support large models
  and very long contexts; training scales with **USP (Ulysses × Ring)**, built from each family's diffusers `_cp_plan` —
  or a self-written one — for agile model integration.



## Supported models

Each model name links to its recipe page. Every documented recipe is labeled with a
[recipe verification level](/diffusion/user-guide/recipe-verification).


| Model                                                   | Task      | Canonical recipes                         |
| ------------------------------------------------------- | --------- | ----------------------------------------- |
| [Stable Diffusion 3.5](/diffusion/models/sd3/sd3)                 | T2I       | Flow-GRPO + OCR, DiffusionNFT + PickScore |
| [Qwen-Image](/diffusion/models/qwen-image/qwen-image)             | T2I       | Flow-GRPO + PickScore (flow_grpo-aligned) |
| [Wan2.2-T2V-A14B](/diffusion/models/wan/wan2-2)                   | T2V       | Flow-GRPO + PickScore, LoRA SFT           |
| [LTX-2.3](/diffusion/models/ltx/ltx2)                             | T2V       | Flow-GRPO + PickScore                     |
| [Cosmos3 (Edge / Nano / Super)](/diffusion/models/cosmos/cosmos3) | T2I       | Flow-GRPO + PickScore                     |
| [MiniMax H3](/diffusion/models/h3/h3)                             | T2VA      | **Not merged** — [PR #154](https://github.com/radixark/miles_diffusion/pull/154); 2-GPU recipe; large-scale coming soon |




## Feature support matrix

- ✅ **Recipe-backed** — exercised by a canonical recipe in `scripts/` or a CI test.
- 🟡 **Implemented** — the code path exists, but no shipped recipe or test covers this combination yet.
- ❌ **Not supported** — no working code path for this combination today.


|                                          | SD3.5 | Qwen-Image | Wan2.2 | LTX-2.3 | Cosmos3 |
| ---------------------------------------- | ----- | ---------- | ------ | ------- | ------- |
| Flow-GRPO (`policy_loss`)                | ✅     | ✅          | ✅      | ✅       | ✅       |
| DiffusionNFT (`nft`)                     | ✅     | 🟡         | 🟡     | 🟡      | 🟡      |
| SFT (`sft_loss`, `--train-only`)         | 🟡    | 🟡         | ✅      | 🟡      | 🟡      |
| LoRA + IPC weight sync                   | ✅     | ✅          | ✅      | 🟡    | ✅       |
| Single-prompt multi-gen (microgroup > 1) | ✅     | ✅          | ✅      | ❌       | ❌      |
| USP sequence parallelism                 | ❌     | ❌          | ✅    | ❌       | ❌       |
| Deterministic mode                       | ✅     | ✅          | ✅      | ✅       | ❌       |



## Start here

1. **[Installation](/diffusion/getting-started/installation)** — Docker image, pinned dependency versions, bare-metal setup.
2. **[Quick Start](/diffusion/getting-started/quick-start)** — a working Flow-GRPO run on SD3.5 with 2 GPUs.
3. **[Core Concepts](/diffusion/user-guide/concepts)** — the four objects in every miles-diffusion job and the loop that connects
   them.
4. **[Launch Scripts](/diffusion/user-guide/launch-script)** — every argument group in a launch script, annotated.
5. **[Rewards](/diffusion/user-guide/rewards)** — built-in reward models and custom reward hooks.
6. **Model guides** — per-model config and recipes, starting from the [supported models](#supported-models) table above.



## Contribute

- GitHub: [github.com/radixark/miles_diffusion](https://github.com/radixark/miles_diffusion)
- Miles (LLM RL): [github.com/radixark/miles](https://github.com/radixark/miles)

