"""Expand SDK image chunks with the served model's processor, preserving text IDs."""

import base64
import io

import torch
from PIL import Image

from miles.tinker.core.types import UserInputError


def decode_model_input(model_input: dict, processor=None) -> tuple[list[int], dict | None, list[str]]:
    tokens, images = [], []
    tensor_parts = {}
    for chunk in model_input["chunks"]:
        kind = chunk.get("type")
        if kind == "encoded_text":
            tokens.extend(chunk["tokens"])
            continue
        if kind != "image":
            raise UserInputError(f"unsupported model_input chunk type: {kind}")
        if processor is None or not hasattr(processor, "image_token"):
            raise UserInputError("the served model does not support image chunks")
        try:
            data = chunk["data"]
            if isinstance(data, str):
                data = base64.b64decode(data, validate=True)
            with Image.open(io.BytesIO(data)) as image:
                output = processor(
                    text=processor.image_token,
                    images=[image.convert("RGB")],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
        except (OSError, ValueError) as error:
            raise UserInputError(f"invalid image chunk: {error}") from error
        image_tokens = output["input_ids"][0].tolist()
        expected = chunk.get("expected_tokens")
        if expected is not None and expected != len(image_tokens):
            raise UserInputError(f"image expected_tokens={expected}, but the processor produced {len(image_tokens)}")
        tokens.extend(image_tokens)
        for key, value in output.items():
            if key not in ("input_ids", "attention_mask", "mm_token_type_ids"):
                tensor_parts.setdefault(key, []).append(value)
        images.append(f"data:image/{chunk['format']};base64,{base64.b64encode(data).decode('ascii')}")
    multimodal_inputs = {key: torch.cat(parts, dim=0) for key, parts in tensor_parts.items()} or None
    return tokens, multimodal_inputs, images
