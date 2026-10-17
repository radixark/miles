"""The trainer and SGLang processors must expand the same request into matching tokens and pixels."""

import base64
import io
import os

import numpy as np
import pytest
import torch
from PIL import Image

from miles.utils.processing_utils import encode_image_for_rollout_engine, load_processor, load_tokenizer

step3_vl = pytest.importorskip("sglang.srt.multimodal.processors.step3_vl")
sglang_processor_utils = pytest.importorskip("sglang.srt.utils.hf_transformers.processor")

CHECKPOINT = os.environ.get("STEP5_HF_CHECKPOINT")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="set STEP5_HF_CHECKPOINT to a Step5 checkpoint directory")

# (width, height): no crops; one row of two crops; a 3x3 grid with row ends; one column of 300-pixel crops.
IMAGE_SIZES = {"small_square": (256, 256), "wide": (1000, 300), "large": (1600, 1200), "tall": (300, 700)}
# SGLang resizes on the GPU when it has one, we on the CPU; uniform noise is the worst case for the resize.
PIXEL_TOLERANCE = 5e-4


@pytest.fixture(scope="module")
def processors():
    ours = load_processor(CHECKPOINT)
    sglang_tokenizer = sglang_processor_utils.get_processor(CHECKPOINT, trust_remote_code=True)
    return ours, step3_vl.Step3VLProcessor(config=None, tokenizer=sglang_tokenizer)


def _noise_image(size: tuple[int, int]) -> Image.Image:
    width, height = size
    pixels = np.random.RandomState(width * height).randint(0, 256, size=(height, width, 4), dtype=np.uint8)
    return Image.fromarray(pixels, "RGBA")


def _prompt(image: Image.Image) -> str:
    tokenizer = load_tokenizer(CHECKPOINT, trust_remote_code=True)
    content = [
        {"type": "text", "text": "Find x.\n\n"},
        {"type": "image", "image": image},
        {"type": "text", "text": "?"},
    ]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def _as_sglang_loads(image: Image.Image) -> Image.Image:
    payload = encode_image_for_rollout_engine(image).split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _sglang_output(sglang, request_ids: list[int], image: Image.Image):
    text = sglang.tokenizer.decode(request_ids)
    return sglang(
        text=[text], images=[_as_sglang_loads(image)], return_tensors="pt", padding=True, add_special_tokens=False
    )


def _max_abs_diff(ours: torch.Tensor, theirs: torch.Tensor) -> float:
    assert ours.shape == theirs.shape, f"{tuple(ours.shape)} vs {tuple(theirs.shape)}"
    return float((ours - theirs.cpu().float()).abs().max()) if ours.numel() else 0.0


def test_sglang_keeps_its_retokenized_prompt():
    """SGLang skips re-tokenizing only when the processor counts image tokens itself; we mirror the re-tokenized one."""
    assert not hasattr(step3_vl.Step3VLProcessor, "_get_num_multimodal_tokens")


@pytest.mark.parametrize("name", IMAGE_SIZES)
def test_matches_sglang(processors, name):
    ours, sglang = processors
    image = _noise_image(IMAGE_SIZES[name])
    out = ours(text=_prompt(image), images=[image])
    theirs = _sglang_output(sglang, out["input_ids"][0], image)

    assert out["step5_expanded_prompt_ids"].tolist() == theirs["input_ids"].flatten().tolist()
    assert out["step5_num_patches"].tolist() == theirs["num_patches"].tolist()
    overview_diff = _max_abs_diff(out["step5_pixel_values"], theirs["pixel_values"])
    crops = theirs.get("patch_pixel_values", torch.empty((0, 3, 504, 504)))
    crops_diff = _max_abs_diff(out["step5_patch_pixel_values"], crops)
    print(
        f"\n{name} {IMAGE_SIZES[name]}: {len(out['step5_expanded_prompt_ids'])} prompt tokens, "
        f"{int(out['step5_num_patches'].sum())} crops, max |overview diff| {overview_diff:.3g}, "
        f"max |crop diff| {crops_diff:.3g} (SGLang on {theirs['pixel_values'].device})"
    )
    assert overview_diff <= PIXEL_TOLERANCE
    assert crops_diff <= PIXEL_TOLERANCE
