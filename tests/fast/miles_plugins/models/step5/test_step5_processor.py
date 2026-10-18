"""Step5TrainProcessor: SGLang's crop geometry, the expanded prompt layout, the request it sends, and media loading.

A byte-level tokenizer with Step5's special tokens and an automatic BOS stands in for the checkpoint's;
tests/manual/step5/test_processor_parity.py checks the real tokenizer against SGLang itself.
"""

import base64
import io

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from miles.utils.types import STEP5_EXPANDED_PROMPT_KEY
from miles_plugins.models.step5.processor import (
    CROP_NUM_TOKENS,
    CROP_SIZE,
    OVERVIEW_NUM_TOKENS,
    OVERVIEW_SIZE,
    Step5TrainProcessor,
    split_image,
)

_BOS = "<bos>"
_SPECIAL_TOKENS = [
    _BOS,
    "<im_start>",
    "<im_patch>",
    "<im_end>",
    "<patch_start>",
    "<patch_newline>",
    "<patch_end>",
    "<|im_start|>",
    "<|im_end|>",
]


def _step5_like_tokenizer() -> PreTrainedTokenizerFast:
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    backend = Tokenizer(models.BPE(vocab={char: index for index, char in enumerate(sorted(alphabet))}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(_SPECIAL_TOKENS)
    bos_id = backend.token_to_id(_BOS)
    backend.post_processor = processors.TemplateProcessing(single=f"{_BOS} $A", special_tokens=[(_BOS, bos_id)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token=_BOS)


@pytest.fixture(scope="module")
def tokenizer():
    return _step5_like_tokenizer()


@pytest.fixture(scope="module")
def processor(tokenizer):
    return Step5TrainProcessor(tokenizer)


def _image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (40, 80, 120))


def _prompt(num_images: int) -> str:
    return _BOS + "<|im_start|>user\nFind x.\n\n" + "<im_patch>" * num_images + "Why?<|im_end|>\n"


def _expected_block(tokenizer, row_ends: list[bool]) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids
    block = []
    for row_end in row_ends:
        block += [ids("<patch_start>")] + [ids("<im_patch>")] * CROP_NUM_TOKENS + [ids("<patch_end>")]
        block += [ids("<patch_newline>")] if row_end else []
    return block + [ids("<im_start>")] + [ids("<im_patch>")] * OVERVIEW_NUM_TOKENS + [ids("<im_end>")]


class TestSplitImage:
    @pytest.mark.parametrize(
        ("size", "crop_side", "row_ends"),
        [
            ((256, 256), None, []),
            # long side <= 728 but an aspect ratio over 1.5: one column of windows as wide as the short side
            ((300, 700), 300, [True, True, False]),
            # one row of two 504 windows over a 1008 x 300 canvas
            ((1000, 300), CROP_SIZE, [False, False]),
            # 3 x 3 windows over a 1512 x 1512 canvas; every row but the last ends in a newline
            ((1600, 1200), CROP_SIZE, [False, False, True, False, False, True, False, False, False]),
        ],
    )
    def test_crops_follow_sglang_geometry(self, size, crop_side, row_ends):
        split = split_image(_image(*size))
        assert split.row_ends == row_ends
        assert split.overview.size == size
        assert all(crop.size == (crop_side, crop_side) for crop in split.crops)

    def test_a_sliver_is_padded_square_first(self):
        split = split_image(_image(20, 100))
        assert split.overview.size == (100, 100)
        assert split.crops == []

    def test_the_long_side_is_capped_at_3024(self):
        split = split_image(_image(4000, 1000))
        assert split.overview.size == (3024, 756)
        # 6 x 2 windows over a 3024 x 1008 canvas
        assert len(split.crops) == 12
        assert split.row_ends.index(True) == 5


class TestCall:
    @pytest.mark.parametrize("size", [(256, 256), (1000, 300), (1600, 1200)])
    def test_expanded_prompt_replaces_the_placeholder_with_every_crop_then_the_overview(
        self, tokenizer, processor, size
    ):
        image = _image(*size)
        out = processor(text=_prompt(1), images=[image])

        request_ids = out["input_ids"][0]
        placeholder = request_ids.index(tokenizer.convert_tokens_to_ids("<im_patch>"))
        block = _expected_block(tokenizer, split_image(image).row_ends)
        expected = [tokenizer.bos_token_id] + request_ids[:placeholder] + block + request_ids[placeholder + 1 :]
        assert out[STEP5_EXPANDED_PROMPT_KEY].tolist() == expected

    def test_positions_are_the_image_tokens_in_feature_order(self, tokenizer, processor):
        out = processor(text=_prompt(2), images=[_image(1000, 300), _image(256, 256)])

        expanded = out[STEP5_EXPANDED_PROMPT_KEY]
        positions = out["step5_image_positions"]
        assert positions.dtype == torch.int64
        assert (
            positions.tolist()
            == (expanded == tokenizer.convert_tokens_to_ids("<im_patch>")).nonzero().flatten().tolist()
        )
        assert len(positions) == 2 * OVERVIEW_NUM_TOKENS + 2 * CROP_NUM_TOKENS
        assert out["step5_num_patches"].tolist() == [2, 0]

    def test_pixel_tensors_hold_every_overview_and_crop(self, processor):
        out = processor(text=_prompt(2), images=[_image(1600, 1200), _image(256, 256)])

        assert out["step5_pixel_values"].shape == (2, 3, OVERVIEW_SIZE, OVERVIEW_SIZE)
        assert out["step5_patch_pixel_values"].shape == (9, 3, CROP_SIZE, CROP_SIZE)
        assert out["step5_pixel_values"].dtype == out["step5_patch_pixel_values"].dtype == torch.float32

    def test_an_image_without_crops_still_carries_an_empty_crop_tensor(self, processor):
        out = processor(text=_prompt(1), images=[_image(256, 256)])
        assert out["step5_patch_pixel_values"].shape == (0, 3, CROP_SIZE, CROP_SIZE)

    def test_the_request_leaves_without_the_bos_sglang_writes_back(self, tokenizer, processor):
        out = processor(text=_prompt(1), images=[_image(256, 256)])

        assert out["input_ids"][0][0] != tokenizer.bos_token_id
        assert out[STEP5_EXPANDED_PROMPT_KEY][:2].tolist().count(tokenizer.bos_token_id) == 1

    def test_the_image_count_must_match_the_placeholders(self, processor):
        with pytest.raises(AssertionError, match="2 <im_patch> but 1 image"):
            processor(text=_prompt(2), images=[_image(256, 256)])

    def test_a_message_list_is_refused(self, processor):
        with pytest.raises(AssertionError, match="--apply-chat-template"):
            processor(text=[{"role": "user", "content": "hi"}], images=None)

    def test_a_text_prompt_is_left_to_the_tokenizer_path(self, processor):
        with pytest.raises(AssertionError, match="image prompts only"):
            processor(text=_prompt(0), images=None)


class TestExtractMedia:
    def test_loads_every_image_part_at_its_own_size(self, tmp_path, processor):
        image = Image.new("RGBA", (37, 23), (1, 2, 3, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        path = tmp_path / "image.png"
        path.write_bytes(buffer.getvalue())
        data_url = "data:image/None;base64," + base64.b64encode(buffer.getvalue()).decode()
        parts = [image, data_url, {"bytes": buffer.getvalue(), "path": None}, str(path), f"file://{path}"]
        prompt = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}] + [{"type": "image", "image": p} for p in parts],
            }
        ]

        media = processor.extract_media(prompt)

        assert media["videos"] is None
        assert [loaded.size for loaded in media["images"]] == [(37, 23)] * len(parts)

    def test_a_text_prompt_has_no_media(self, processor):
        assert processor.extract_media([{"role": "user", "content": "just text"}]) == {"images": None, "videos": None}

    def test_urls_are_resolved_upstream(self, processor):
        prompt = [{"role": "user", "content": [{"type": "image", "image": "https://example.com/a.png"}]}]
        with pytest.raises(ValueError, match="resolve URLs upstream"):
            processor.extract_media(prompt)
