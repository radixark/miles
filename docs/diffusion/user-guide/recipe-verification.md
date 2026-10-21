---
title: Recipe Verification Levels
description: Evidence levels for documented training recipes.
---
# Recipe verification levels

Every recipe is assigned one evidence level. The level applies to the exact script and
topology named in the model guide, not to the model family as a whole.

<a id="fg"></a>
## 🛡️ FG — Fully gated

- A complete training curve has been run.
- A deterministic E2E runs the canonical recipe itself at least nightly.
- Every registered metric matches the committed standard exactly.

<a id="pg"></a>
## 🧩 PG — Proxy gated

- A complete training curve has been run.
- A deterministic E2E runs a documented, scaled single-node proxy at least nightly.
- Every registered metric matches the committed standard exactly.

<a id="v"></a>
## 📈 V — Verified

A complete training curve has been run, but no deterministic E2E satisfies the FG or PG criteria above.

<a id="nv"></a>
## ○ NV — Not verified

No complete training curve has been run. Smoke tests and short debugging runs do not
count as verification.

## Current levels

- **🛡️ FG**
  - `run_diffusion_grpo_sd3_ocr_sglang.py` — SD3.5 Flow-GRPO + OCR.
  - `run_diffusion_grpo_ltx23_sglang.py` — LTX-2.3 Flow-GRPO + PickScore.
  - `run_diffusion_nft_sd3_pickscore.py` — SD3.5 DiffusionNFT + PickScore.
- **🧩 PG**
  - `run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py` — Wan2.2 17-GPU
    full-finetune Flow-GRPO + PickScore.
- **📈 V**
  - `run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py` — Qwen-Image
    Flow-GRPO + PickScore.
  - `run_diffusion_grpo_cosmos3_pickscore_t2i_4gpu.py` — Cosmos3-Nano
    Flow-GRPO + PickScore.
  - `run_diffusion_grpo_h3_t2va_2gpu.py` — MiniMax H3 Flow-GRPO + PickScore.
- **○ NV**
  - `run_diffusion_grpo_wan22_pickscore_5gpu.py` — Wan2.2 5-GPU LoRA
    Flow-GRPO + PickScore.
  - `run_diffusion_sft_wan22.py` — Wan2.2 4-GPU LoRA SFT.
