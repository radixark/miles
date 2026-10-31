"""Through a live engine: a prompt with a split word and one image, sent with
one image placeholder per image, runs id for id as the expanded prompt the
training record holds. Needs an SGLang server for a Qwen VL checkpoint at
MILES_E2E_SGLANG_URL whose Qwen loader honours client input_ids
(sgl-project/sglang exact-token path), and the checkpoint at
MILES_E2E_HF_CHECKPOINT for the processor.

    MILES_E2E_SGLANG_URL=http://127.0.0.1:30000 MILES_E2E_HF_CHECKPOINT=/path/Qwen3.8-27B \\
        pytest tests/e2e/sglang/test_generate_one_pad_per_image.py
"""

import base64
import io
import os
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from miles.rollout.generate_utils.generate_endpoint_utils import (
    collapse_image_pads,
    compute_prompt_ids_from_sample,
    image_pad_id_for,
)
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

URL = os.environ.get("MILES_E2E_SGLANG_URL")
CKPT = os.environ.get("MILES_E2E_HF_CHECKPOINT")
pytestmark = pytest.mark.skipif(not (URL and CKPT), reason="MILES_E2E_SGLANG_URL and MILES_E2E_HF_CHECKPOINT not set")


def _png(width=1024, height=576) -> Image.Image:
    image = Image.new("RGB", (width, height), (40, 80, 120))
    for x in range(0, width, 64):
        for y in range(0, height, 64):
            if (x // 64 + y // 64) % 2 == 0:
                image.paste((200, 200, 60), (x, y, x + 32, y + 32))
    return image


def test_one_pad_per_image_runs_the_expanded_ids_id_for_id():
    tokenizer = load_tokenizer(CKPT, trust_remote_code=True)
    processor = load_processor(CKPT, trust_remote_code=True)
    state = SimpleNamespace(tokenizer=tokenizer, processor=processor)
    pad = image_pad_id_for(state)
    assert pad is not None
    image = _png()
    # "D" + "escribe": a split word the engine would re-tokenize differently
    prompt = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>D"
        "escribe the picture.<|im_end|>\n<|im_start|>assistant\n"
    )
    sample = Sample(prompt=prompt, multimodal_inputs={"images": [image]})
    expanded = list(compute_prompt_ids_from_sample(state, sample))
    n_pads = sum(1 for t in expanded if t == pad)
    assert n_pads > 1, "the processor expands the placeholder to the image's token count"
    wire = collapse_image_pads(expanded, pad)
    assert sum(1 for t in wire if t == pad) == 1

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    body = {
        "input_ids": wire,
        "image_data": ["data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    reply = requests.post(URL.rstrip("/") + "/generate", json=body, timeout=600).json()
    meta = reply["meta_info"]
    ran = [item[1] for item in meta["input_token_logprobs"]]
    assert len(ran) == len(expanded), (len(ran), len(expanded))
    # the engine reports the image placeholders as 0; every other id must match
    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(ran, expanded, strict=True)) if b != pad and a != b]
    assert mismatches == [], mismatches[:5]
    assert meta["prompt_tokens"] == len(expanded)
