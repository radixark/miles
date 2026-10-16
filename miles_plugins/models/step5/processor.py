"""Mirror SGLang's Step3-VL image preprocessing and expanded prompt on CPU.

Each request image becomes crop and overview blocks. `Step5TrainProcessor` returns
pixels and `step5_expanded_prompt_ids` so training uses the prompt SGLang sampled from.
"""

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from miles.utils.types import STEP5_EXPANDED_PROMPT_KEY

# The geometry and tokens of SGLang's Step3VLProcessor (sglang/srt/multimodal/processors/step3_vl.py).
OVERVIEW_SIZE = 728
CROP_SIZE = 504
OVERVIEW_NUM_TOKENS = 169
CROP_NUM_TOKENS = 81
IMAGE_TOKEN = "<im_patch>"
_OVERVIEW_START, _OVERVIEW_END = "<im_start>", "<im_end>"
_CROP_START, _CROP_END, _CROP_ROW_END = "<patch_start>", "<patch_end>", "<patch_newline>"
_MAX_LONG_SIDE = 3024
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class SplitImage:
    """One image cut the way SGLang's ``ImagePatcher`` cuts it."""

    overview: Image.Image
    crops: list[Image.Image]
    # One flag per crop: a <patch_newline> follows it (the last crop of every row but the final one).
    row_ends: list[bool]


def split_image(image: Image.Image) -> SplitImage:
    """Overview and crops of an RGB image; crops only for a long side over 728 or an aspect ratio over 1.5."""
    width, height = image.size
    if _needs_square_pad(width, height):
        image = _square_pad(image)
        width, height = image.size
    capped_width, capped_height = _cap_long_side(width, height)
    overview = image.resize((capped_width, capped_height), Image.Resampling.BILINEAR)
    window = _window_size(long_side=max(capped_width, capped_height), short_side=min(capped_width, capped_height))
    if window == 0:
        return SplitImage(overview=overview, crops=[], row_ends=[])

    canvas_size = _crop_canvas_size(capped_width, capped_height, window)
    # SGLang compares the canvas against the pre-cap size, which only differs past the 3,024 cap.
    canvas = overview if canvas_size == (width, height) else overview.resize(canvas_size, Image.Resampling.BILINEAR)
    columns = _window_starts(canvas_size[0], window)
    rows = _window_starts(canvas_size[1], window)
    crops = [canvas.crop((x, y, x + window, y + window)) for y in rows for x in columns]
    row_ends = [(index + 1) % len(columns) == 0 and index != len(crops) - 1 for index in range(len(crops))]
    return SplitImage(overview=overview, crops=crops, row_ends=row_ends)


def to_pixel_values(images: list[Image.Image], size: int) -> torch.Tensor:
    """float32 ``[len(images), 3, size, size]``: to-tensor, CLIP normalization, then an antialiased resize."""
    if not images:
        return torch.empty((0, 3, size, size), dtype=torch.float32)
    return torch.stack([_normalize_and_resize(image, size) for image in images])


def _image_block(split: SplitImage) -> str:
    """The text SGLang puts in place of one image's ``<im_patch>``: every crop, then the overview."""
    crop = _CROP_START + IMAGE_TOKEN * CROP_NUM_TOKENS + _CROP_END
    crops = "".join(crop + (_CROP_ROW_END if row_end else "") for row_end in split.row_ends)
    return crops + _OVERVIEW_START + IMAGE_TOKEN * OVERVIEW_NUM_TOKENS + _OVERVIEW_END


class Step5TrainProcessor:
    """Duck-typed processor for miles' multimodal rollout path (Step5 VL)."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        # SGLang re-tokenizes the decoded request with the tokenizer's own special tokens (its Step3 processor
        # drops add_special_tokens=False), so requests leave without the BOS the chat template wrote.
        self._retokenize_prefix = tokenizer("")["input_ids"]

    def extract_media(self, prompt: list[dict[str, Any]]) -> dict[str, Any]:
        images = [
            _load_image(part["image"])
            for message in prompt
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        return {"images": images or None, "videos": None}

    def __call__(self, text=None, images=None, videos=None, audios=None, **kwargs) -> dict[str, Any]:
        assert not videos and not audios, "the Step5 processor takes images only"
        assert isinstance(text, str), (
            "the Step5 processor expects the chat-templated prompt (run with --apply-chat-template), "
            f"got {type(text).__name__}"
        )
        # SGLang re-tokenizes only requests that carry images; text prompts take the tokenizer path.
        assert images, "the Step5 processor handles image prompts only"
        request_ids = self._request_ids(text)
        splits = [split_image(_to_rgb(image)) for image in images]
        assert request_ids.count(self._image_token_id) == len(splits), (
            f"the prompt holds {request_ids.count(self._image_token_id)} {IMAGE_TOKEN} but {len(splits)} "
            "image(s) came with it"
        )
        expanded_prompt_ids = self._expand_as_sglang(request_ids, splits)
        return {
            "input_ids": [request_ids],
            "step5_pixel_values": to_pixel_values([split.overview for split in splits], OVERVIEW_SIZE),
            "step5_patch_pixel_values": to_pixel_values([crop for split in splits for crop in split.crops], CROP_SIZE),
            "step5_num_patches": torch.tensor([len(split.crops) for split in splits], dtype=torch.int64),
            "step5_image_positions": (expanded_prompt_ids == self._image_token_id).nonzero().flatten(),
            STEP5_EXPANDED_PROMPT_KEY: expanded_prompt_ids,
        }

    def _request_ids(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        prefix = self._retokenize_prefix
        return ids[len(prefix) :] if prefix and ids[: len(prefix)] == prefix else ids

    def _expand_as_sglang(self, request_ids: list[int], splits: list[SplitImage]) -> torch.Tensor:
        pieces = self._tokenizer.decode(request_ids).split(IMAGE_TOKEN)
        assert len(pieces) == len(splits) + 1, f"{len(pieces) - 1} {IMAGE_TOKEN} in the decoded prompt"
        expanded_text = pieces[0] + "".join(
            _image_block(split) + piece for split, piece in zip(splits, pieces[1:], strict=True)
        )
        return torch.tensor(self._tokenizer(expanded_text)["input_ids"], dtype=torch.int64)


def _needs_square_pad(width: int, height: int) -> bool:
    ratio = width / height
    return min(width, height) < 32 and (ratio > 4 or ratio < 1 / 4)


def _square_pad(image: Image.Image) -> Image.Image:
    size = max(image.size)
    padded = Image.new(image.mode, (size, size), 0)
    padded.paste(image, (0, 0))
    return padded


def _cap_long_side(width: int, height: int) -> tuple[int, int]:
    if max(width, height) <= _MAX_LONG_SIDE:
        return width, height
    scale = _MAX_LONG_SIDE / max(width, height)
    return int(width * scale), int(height * scale)


def _window_size(*, long_side: int, short_side: int) -> int:
    """The square crop side, or 0 for no crops."""
    if long_side <= OVERVIEW_SIZE:
        return short_side if long_side / short_side > 1.5 else 0
    return min(short_side, CROP_SIZE) if long_side / short_side > 4 else CROP_SIZE


def _crop_canvas_size(width: int, height: int, window: int) -> tuple[int, int]:
    return _crop_canvas_side(width, window), _crop_canvas_side(height, window)


def _crop_canvas_side(length: int, window: int) -> int:
    ratio = length / window
    if ratio < 1:
        return length
    num_windows = int(ratio) + 1 if ratio - length // window > 0.2 else int(ratio)
    return window * num_windows


def _window_starts(length: int, window: int) -> list[int]:
    count = 1 if length <= window else math.ceil((length - window) / window + 1)
    starts = [window * index for index in range(count)]
    if len(starts) > 1 and starts[-1] + window > length:
        starts[-1] = length - window
    return starts


def _normalize_and_resize(image: Image.Image, size: int) -> torch.Tensor:
    pixels = TF.normalize(TF.to_tensor(image), mean=_CLIP_MEAN, std=_CLIP_STD)
    return TF.resize(pixels, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)


def _to_rgb(image: Image.Image) -> Image.Image:
    # The same conversion encode_image_for_rollout_engine applies before SGLang ever sees the image.
    return image if image.mode == "RGB" else image.convert("RGB")


def _load_image(spec: Any) -> Image.Image:
    """One image content part as a PIL image at its own size (no resize)."""
    if isinstance(spec, Image.Image):
        image = spec
    elif isinstance(spec, (bytes, bytearray)):
        image = Image.open(io.BytesIO(spec))
    elif isinstance(spec, dict) and spec.get("bytes") is not None:
        image = Image.open(io.BytesIO(spec["bytes"]))
    elif isinstance(spec, str) and spec.startswith("data:"):
        image = Image.open(io.BytesIO(base64.b64decode(spec.split(",", 1)[1])))
    elif isinstance(spec, str) and not spec.startswith(("http://", "https://")):
        image = Image.open(Path(spec.removeprefix("file://")))
    else:
        raise ValueError(f"unsupported Step5 image payload: {str(spec)[:64]!r} (resolve URLs upstream)")
    image.load()
    return image
