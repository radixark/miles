---
title: Rewards
description: Built-in reward models (PickScore, OCR), rm_hub dispatch, and prompt data format.
---
Miles-diffusion scores generated images (or video frames) after each rollout
microgroup. Reward computation lives in `miles/rollout/rm_hub/` and is invoked
from `sglang_diffusion_rollout.generate_and_rm_microgroup`.

For `--custom-rm-path`, `--custom-reward-post-process-path`, and other
`--*-path` hooks, see [Customization](/diffusion/user-guide/customization).

## 1. At a glance

| Stage | Flag | Role |
|---|---|---|
| Reward type | `--rm-type` | Selects built-in scorer (`pickscore`, `ocr`) |
| Per-sample override | `metadata.rm_type` in JSONL | Overrides global `--rm-type` |
| Custom reward / norm | see [Customization](/diffusion/user-guide/customization) | `--custom-rm-path`, `--custom-reward-post-process-path` |

## 2. Built-in reward models

### PickScore (`--rm-type pickscore`)

Implementation: `miles/rollout/rm_hub/pickscore.py`.

PickScore scores text–image alignment using a CLIP model pair:

- Processor: `--pickscore-processor-path` (e.g.
  `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`)
- Model: `--pickscore-model-path` (e.g. `yuvalkirstain/PickScore_v1`)

Scoring formula:

```
score = exp(logit_scale) * dot(text_emb, image_emb) / 26.0
```

The `/ 26.0` scaling maps raw PickScore logits (~0–26) to roughly 0–1.

PickScore runs as a **Ray actor pool** (`PickScoreRewardActor`) with round-robin
batching. For video outputs, frames are uniformly sampled
(`--pickscore-num-frames`) and scores are averaged.

| Flag | Default | Description |
|---|---|---|
| `--pickscore-num-workers` | 1 | Ray actor count |
| `--pickscore-num-gpus-per-worker` | 1.0 | GPU per worker (non-colocate) |
| `--pickscore-batch-size` | 8 | Batch size per actor |
| `--pickscore-processor-path` | — | Required for pickscore |
| `--pickscore-model-path` | — | Required for pickscore |
| `--pickscore-num-frames` | None | Video frame sampling count |
| `--colocate-reward` | False | Share rollout GPUs (0.05 GPU/worker) |

Example from `scripts/run_diffusion_nft_sd3_pickscore.py`:

```bash
--rm-type pickscore \
--pickscore-num-workers 1 \
--pickscore-num-gpus-per-worker 1.0 \
--pickscore-batch-size 8 \
--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
--pickscore-model-path yuvalkirstain/PickScore_v1
```

### OCR (`--rm-type ocr`)

Implementation: `miles/rollout/rm_hub/ocr.py`.

OCR reward compares PaddleOCR output against target text embedded in the
prompt. The target is the string between the first pair of double quotes.
Simplified scoring:

```python
target = prompt.split('"')[1]
reward = 1 - levenshtein_distance(recognized, target) / len(target)
```

The implementation also lowercases both sides, strips spaces, treats a
substring hit as perfect (`dist=0`), and caps `dist` at `len(target)` — see
`miles/rollout/rm_hub/ocr.py`.

OCR runs on **CPU** Ray actors (`--ocr-num-workers`, default 4). Used by the
SD3 Flow-GRPO recipe (`scripts/run_diffusion_grpo_sd3_ocr_sglang.py`).

### Remote RM (`--rm-type remote_rm`)

The CLI exposes `--rm-url` for a remote reward service, but **`rm_hub` does not
implement `remote_rm` today** — selecting it raises `NotImplementedError`.
Use `--custom-rm-path` to call an external service instead (see below).

## 3. Call chain

```
generate_and_rm_microgroup()
  → batched_async_rm(args, microgroup)          # rm_hub/__init__.py
    → custom_rm_path?  user batched function
    → all pickscore?   pickscore_rm (batched)
    → else             per-sample async_rm → ocr / pickscore / NotImplementedError
  → sample.reward = score
  → RolloutManager._post_process_rewards()      # GRPO advantage normalization
```

After rollout, `RolloutManager._post_process_rewards` subtracts the mean and
optionally divides by std to produce normalized advantages for training.
Override that path with `--custom-reward-post-process-path` — see
[Customization](/diffusion/user-guide/customization) § Reward.

## 4. Prompt data

### JSONL format

Training prompts are loaded from `.jsonl` files via `miles/utils/diffusion_data.py`:

```json
{"input": "A photo of a cat wearing sunglasses"}
{"input": "A logo saying \"Miles\"", "metadata": {"rm_type": "ocr"}}
```

| Field | CLI mapping | Notes |
|---|---|---|
| Prompt text | `--input-key input` | Required non-empty string |
| Per-sample metadata | `--metadata-key metadata` | Optional dict; supports `rm_type` override |

### Dataset subsets

Dataset repo: [`rockdu/miles-diffusion-datasets`](https://huggingface.co/datasets/rockdu/miles-diffusion-datasets)

| Subset | Used by |
|---|---|
| `flowgrpo_pickscore/` | PickScore recipes (SD3 NFT, Qwen-Image, Wan2.2, LTX) |
| `flowgrpo_ocr/` | SD3 OCR Flow-GRPO, NFT smoke test |

### Per-sample rm_type override

JSONL `metadata.rm_type` overrides the global `--rm-type` for that sample:

```python
# rm_hub/__init__.py
metadata.get("rm_type") or args.rm_type
```

Mixed rm_types within one microgroup fall back to per-sample dispatch (no
batched PickScore fast path).
