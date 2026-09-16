---
title: Miles-Diffusion Documentation
---
[Miles-diffusion](https://github.com/radixark/miles_diffusion) is currently a standalone repository built on
[Miles](https://github.com/radixark/miles)' design philosophy, focused on RL post-training for image and video diffusion
models. [sglang-diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen) serves the
rollout, and the DiT trains under **FSDP2** on a backend that co-evolves with Miles' own. Models load from a diffusers
pipeline, or from a native package when a family brings its own modeling. Shipped recipes carry explicit
[verification levels](/diffusion/user-guide/recipe-verification). Custom rewards, losses, and rollout functions plug in through flags.

## Core features

- **Verified Recipes for Latest Diffusion Models.** Launchers for Wan2.2-T2V-A14B, Qwen-Image,
  LTX-2.3, Cosmos3-Nano, and SD3.5. `TrainPipelineConfig` allows for easy model support.
- **Quality control on three fronts.** Deterministic mode supports bit-for-bit comparisons for recipes covered by
  committed E2E standards; sglang-side monkey patches reduce train/rollout mismatches; and an FSDP2 param-dtype patch
  provides per-parameter fp32 control under the mixed-precision policy. See [Deterministic
  Training](/diffusion/advanced/deterministic) and [Dtype Control](/diffusion/advanced/dtype-control).
- **SFT, DiffusionNFT, and Flow-GRPO under one trainer.** The loss type, training-batch preparation, rollout function,
  and reward function are all **replaceable components**, so integrating a new algorithm — or swapping in your own
  customized component — is easy.
- **Sglang native.** Rollout runs **on the inference engine itself** — the sglang-diffusion serving stack — with RL
  support and optimizations living engine-side. An optional curated set of monkey patches aligns selected engine
  operations with the training-side forward.
- **Multiple parallelisms.** The rollout engines scale with **tensor and sequence parallelism** to support large models
  and very long contexts; training scales with **USP (Ulysses × Ring)**, built from each family's diffusers `_cp_plan` —
  or a self-written one — for agile model integration.
- **LoRA training support.** With `--lora-ipc-weight-sync`, PEFT LoRA on the FSDP2 actor ships only
  `lora_A`/`lora_B` pairs to colocated rollout engines over CUDA IPC and merges them engine-side. See
  [LoRA Training and Weight Sync](/diffusion/advanced/lora).



## Supported models

Each model links to its recipe page; see
[verification levels](/diffusion/user-guide/recipe-verification). Validated models also
appear in the [Miles model list](/#supported-models).


| Model                                                   | Task      | Canonical recipes                         |
| ------------------------------------------------------- | --------- | ----------------------------------------- |
| [Stable Diffusion 3.5](/diffusion/models/sd3/sd3)                 | T2I       | Flow-GRPO + OCR, DiffusionNFT + PickScore |
| [Qwen-Image](/diffusion/models/qwen-image/qwen-image)             | T2I       | Flow-GRPO + PickScore (flow_grpo-aligned) |
| [Wan2.2-T2V-A14B](/diffusion/models/wan/wan2-2)                   | T2V       | Flow-GRPO + PickScore, LoRA SFT           |
| [LTX-2.3](/diffusion/models/ltx/ltx2)                             | T2V       | Flow-GRPO + PickScore                     |
| [Cosmos3 (Edge / Nano / Super)](/diffusion/models/cosmos/cosmos3) | T2I       | Flow-GRPO + PickScore                     |
| [MiniMax H3](/diffusion/models/h3/h3)                             | T2VA      | **Not merged** — [PR #154](https://github.com/radixark/miles_diffusion/pull/154); 2-GPU PR-only recipe |




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
| Single-prompt multi-gen (microgroup > 1) | ✅     | ✅          | ✅      | 🟡       | ❌      |
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

