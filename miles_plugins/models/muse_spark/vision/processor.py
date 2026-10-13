"""Training-side processor for Muse Spark omni images (Inkling ``InklingTrainProcessor`` pattern).

Image path is a line-for-line port of the omni processor's CPU fallback
(``_image_to_uint8`` / ``_normalization_lut`` / PIL LANCZOS ``_resize_and_normalize``
/ ``preprocess_image``). Token path: every ``<image>`` the chat template rendered
becomes ``<|image_start|>`` + ``<|patch|>`` * n + ``<|image_end|>`` before
tokenization, exactly as SGLang expands it, so the rollout engine's
pretokenized path accepts the ids unchanged and lengths never change on the
trainer (CP-safe).

Outputs (all concatenable along dim 0 by ``data.py``):
  input_ids            [[...]]                 expanded prompt ids
  mm_vision_patches    [sum(gh*gw), T*C*P*P]   bf16 patch rows in NaViT order
  mm_vision_grids      [n_images, 2]           (grid_h, grid_w) per image
  mm_vision_positions  [sum(gh*gw/f^2)]        sample-local indices of the <|patch|> tokens
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from PIL import Image

from miles_plugins.models.muse_spark.vision.navit import MuseSparkVisionConfig, find_resize_target, patchify_clip

IMAGE_PLACEHOLDER = "<image>"
IMAGE_START_TOKEN = "<|image_start|>"
IMAGE_PATCH_TOKEN = "<|patch|>"
IMAGE_END_TOKEN = "<|image_end|>"
_IMAGE_PART_TYPES = frozenset({"image", "input_image", "image_url"})


def image_to_uint8(image: Any) -> torch.Tensor:
    """``[3,H,W]`` uint8, same coercions as SGLang."""
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1)
    if isinstance(image, np.ndarray):
        value = torch.from_numpy(np.array(image, copy=True))
    else:
        value = torch.as_tensor(image).cpu()
    if value.ndim != 3:
        raise ValueError(f"unexpected image shape {tuple(value.shape)}")
    if value.shape[0] in (1, 3, 4):
        value = value[:3]
    else:
        value = value[..., :3].permute(2, 0, 1)
    if value.dtype != torch.uint8:
        if value.is_floating_point() and value.max().item() <= 1.0:
            value = value * 255
        value = value.clamp(0, 255).to(torch.uint8)
    if value.shape[0] == 1:
        value = value.expand(3, -1, -1)
    return value.contiguous()


@lru_cache(maxsize=32)
def normalization_lut(mean: tuple, std: tuple) -> np.ndarray:
    # Preserve the reference float32 operation order for all 256 pixel values.
    values = torch.arange(256, dtype=torch.float32).div_(255.0).clamp_(0.0, 1.0)
    return ((values[None, :] - torch.tensor(mean)[:, None]) / torch.tensor(std)[:, None]).numpy()


def resize_and_normalize(image: Any, target_h: int, target_w: int, mean: tuple, std: tuple) -> torch.Tensor:
    """PIL LANCZOS resize + LUT normalization -> ``[3,H,W]`` fp32 (SGLang's non-Rust path)."""
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(image_to_uint8(image).permute(1, 2, 0).numpy())
    array = np.asarray(pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS))
    lut = normalization_lut(tuple(mean), tuple(std))
    normalized = np.empty((3, target_h, target_w), dtype=np.float32)
    for channel in range(3):
        normalized[channel] = lut[channel, array[:, :, channel]]
    return torch.from_numpy(normalized)


def preprocess_image(image: Any, config: MuseSparkVisionConfig) -> tuple[torch.Tensor, int]:
    """``(fp32 [3,H,W], llm_token_count)`` with H, W multiples of the 28 px token patch."""
    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        image = image_to_uint8(image)
        height, width = image.shape[-2:]
    target_h, target_w, token_count = find_resize_target(
        height, width, config.token_patch_size, config.min_num_image_tokens, config.max_num_image_tokens
    )
    return resize_and_normalize(image, target_h, target_w, config.image_mean, config.image_std), token_count


def _load_pil_image(spec: Any) -> Image.Image:
    """Resolve one image content part's payload to PIL (no resize); URLs must be resolved upstream."""
    if isinstance(spec, Image.Image):
        return spec
    if isinstance(spec, dict):
        spec = spec.get("url", spec.get("path"))
    if isinstance(spec, (bytes, bytearray)):
        import io

        image = Image.open(io.BytesIO(bytes(spec)))
        image.load()
        return image
    if isinstance(spec, str):
        path = spec[len("file://") :] if spec.startswith("file://") else spec
        if path.startswith(("http://", "https://", "data:")):
            raise ValueError(f"Muse Spark extract_media: resolve URL images upstream, got {spec[:64]!r}")
        image = Image.open(path)
        image.load()
        return image
    raise TypeError(f"unsupported image payload type {type(spec).__name__}")


class MuseSparkTrainProcessor:
    """Duck-typed processor for miles' generic multimodal pipeline (Muse Spark omni)."""

    def __init__(self, hf_checkpoint: str, tokenizer=None):
        with open(os.path.join(hf_checkpoint, "config.json"), encoding="utf-8") as handle:
            hf_config = json.load(handle)
        self.vision_config = MuseSparkVisionConfig.from_hf(hf_config)
        if tokenizer is None:
            from miles.utils.processing_utils import load_tokenizer

            tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.image_start_id = self._single_token_id(IMAGE_START_TOKEN)
        self.image_patch_id = self._single_token_id(IMAGE_PATCH_TOKEN)
        self.image_end_id = self._single_token_id(IMAGE_END_TOKEN)

    def _single_token_id(self, literal: str) -> int:
        ids = self.tokenizer.encode(literal, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Muse Spark special {literal!r} must encode to one id, got {ids}")
        return ids[0]

    # -- miles hooks ---------------------------------------------------------

    def extract_media(self, prompt: list[dict[str, Any]]) -> dict:
        images = []
        for message in prompt:
            content = message.get("content", "")
            if not isinstance(content, (list, tuple)):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                    images.append(_load_pil_image(part.get("image", part.get("image_url"))))
        return {"images": images or None, "videos": None, "audios": None}

    def expand_text(self, text: str, token_counts: list[int]) -> str:
        """Replace each ``<image>`` (left to right) by the start/patch-run/end literal, like SGLang."""
        for count in token_counts:
            replacement = IMAGE_START_TOKEN + IMAGE_PATCH_TOKEN * count + IMAGE_END_TOKEN
            index = text.find(IMAGE_PLACEHOLDER)
            if index < 0:
                raise ValueError("fewer <image> placeholders in the prompt than images given")
            text = text[:index] + replacement + text[index + len(IMAGE_PLACEHOLDER) :]
        if IMAGE_PLACEHOLDER in text:
            raise ValueError("more <image> placeholders in the prompt than images given")
        return text

    def __call__(self, text=None, images=None, videos=None, audios=None, **kwargs):
        assert not videos, "Muse Spark train processor: video inputs are not wired yet"
        assert not audios, "Muse Spark train processor: audio is out of scope"
        assert isinstance(text, str), (
            "Muse Spark processor expects the chat-template-rendered prompt string "
            f"(run with --apply-chat-template), got {type(text).__name__}"
        )
        images = list(images or [])
        tensors, counts, grids = [], [], []
        for image in images:
            tensor, count = preprocess_image(image, self.vision_config)
            patches, grid_h, grid_w = patchify_clip(
                tensor.to(torch.bfloat16), self.vision_config.patch_size, self.vision_config.patch_temporal
            )
            assert count == grid_h * grid_w // self.vision_config.downsample_factor**2
            tensors.append(patches)
            counts.append(count)
            grids.append((grid_h, grid_w))
        input_ids = self.tokenizer.encode(self.expand_text(text, counts), add_special_tokens=False)
        out: dict[str, Any] = {"input_ids": [input_ids]}
        if images:
            ids = torch.tensor(input_ids)
            positions = (ids == self.image_patch_id).nonzero(as_tuple=True)[0]
            if positions.numel() != sum(counts):
                raise ValueError(f"{positions.numel()} <|patch|> tokens but images need {sum(counts)}")
            out["mm_vision_patches"] = torch.cat(tensors)
            out["mm_vision_grids"] = torch.tensor(grids, dtype=torch.long)
            out["mm_vision_positions"] = positions.to(torch.long)
        return out
