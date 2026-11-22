---
title: Recipe Verification Levels
description: Evidence levels for documented training recipes.
---

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
  - `run_diffusion_grpo_cosmos3_pickscore_t2i_4gpu.py` — Cosmos3-Nano
    Flow-GRPO + PickScore.
  - `run_diffusion_grpo_h3_t2va_2gpu.py` — MiniMax H3 t2va Flow-GRPO + PickScore.
  - `run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py` — Qwen-Image
    Flow-GRPO + PickScore.
- **🧩 PG**
  - `run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py` — Wan2.2 17-GPU
    full-finetune Flow-GRPO + PickScore.
  - `run_diffusion_nft_krea2_async.py` — Krea-2 8-GPU async DiffusionNFT + OCR.
- **📈 V**
  - `run_diffusion_grpo_qwenimage21_pickscore_1gpu.py` — Qwen-Image 2.1
    Flow-GRPO + PickScore.
  - `run_diffusion_nft_krea2_8gpu.py` — Krea-2 8-GPU colocated DiffusionNFT + OCR.
  - `run_diffusion_nft_krea2.py` — Krea-2 2-GPU colocated DiffusionNFT + OCR.
  - `run_diffusion_nft_flux3_image_pickscore_8gpu.py` — FLUX 3 Image Klein 8-GPU colocated DiffusionNFT + PickScore.
  - `run_diffusion_grpo_flux3_image_hps_8gpu.py` — FLUX 3 Image Klein 8-GPU colocated full-finetune Flow-GRPO + HPSv2.1.
  - `run_diffusion_sft_h3_t2va.py` — MiniMax H3 8-GPU t2va LoRA SFT.
  - `run_diffusion_grpo_sd3_hps_sglang.py` — SD3.5 Flow-GRPO + HPSv2.1.
  - `run_diffusion_grpo_sd3_ocr_pickscore_sglang.py` — SD3.5 Flow-GRPO + 0.8 OCR + 0.2 PickScore.
- **○ NV**
  - `run_diffusion_grpo_wan22_pickscore_5gpu.py` — Wan2.2 5-GPU LoRA
    Flow-GRPO + PickScore.
  - `run_diffusion_sft_wan22.py` — Wan2.2 4-GPU LoRA SFT.
  - `run_diffusion_sft_h3_ref2va.py` — MiniMax H3 8-GPU Ref2VA LoRA SFT.
