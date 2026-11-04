"""Inkling2 image processor for miles' multimodal rollout: SGLang's own renderer and image
preprocessing, so prompts and vision patches match what the engine builds for the same request."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.multimodal.inkling2 import Inkling2ImageProcessor
from sglang.srt.parser.inkling2.server import render_inkling2_messages, rendering_tokenizer
from sglang.srt.parser.inkling_tokenizer import IMAGE_TOKEN_ID

from miles.utils.processing_utils import encode_image_for_rollout_engine, load_tokenizer
from miles_plugins.models.inkling.mm_processor import InklingTrainProcessor


def _renderer_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    # The renderer emits one image placeholder per image_url part; it checks the URL is a
    # nonempty string but never reads it (the pixels come from extract_media).
    parts = [
        {"type": "image_url", "image_url": {"url": "image"}} if part["type"] in ("image", "input_image") else part
        for part in content
    ]
    return {**message, "content": parts}


class Inkling2TrainProcessor:
    def __init__(self, hf_checkpoint: str):
        self.rendering_tokenizer = rendering_tokenizer(load_tokenizer(hf_checkpoint, trust_remote_code=True))
        self.image_processor = Inkling2ImageProcessor.from_pretrained(hf_checkpoint)

    extract_media = InklingTrainProcessor.extract_media

    def __call__(self, text=None, images=None, videos=None, audios=None, **kwargs):
        assert not videos and not audios, "Inkling2 supports text and images only"
        assert isinstance(text, (list, tuple)), (
            "Inkling2 processor expects the raw message list as `text` (run without "
            f"--apply-chat-template so the prompt stays structured), got {type(text).__name__}"
        )
        input_ids = render_inkling2_messages([_renderer_message(m) for m in text], self.rendering_tokenizer)
        images = list(images or [])
        assert input_ids.count(IMAGE_TOKEN_ID) == len(images), (
            f"Inkling2 render produced {input_ids.count(IMAGE_TOKEN_ID)} image placeholder(s) "
            f"but {len(images)} image(s) given"
        )
        out: dict[str, Any] = {"input_ids": [input_ids]}
        if images:
            # The exact bytes the rollout sends the engine, so both decode the same pixels.
            features = self.image_processor.preprocess([encode_image_for_rollout_engine(image) for image in images])
            out["mm_vision_patches"] = features["vision_patches_bthwc"]
            out["mm_vision_grid_thw"] = features["vision_grid_thw"]
            out["mm_vision_num_patches"] = torch.tensor(features["num_tokens"], dtype=torch.long)
        return out
