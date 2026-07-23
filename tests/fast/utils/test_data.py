from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=[])

import pytest
from PIL import Image

from miles.utils.data import filter_long_prompt
from miles.utils.processing_utils import load_processor, load_tokenizer, process_vision_info
from miles.utils.types import Sample

VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(VLM_MODEL_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def processor():
    proc = load_processor(VLM_MODEL_NAME, trust_remote_code=True)
    assert proc is not None
    return proc


def _make_vlm_sample(tokenizer, processor, with_image: bool) -> Sample:
    # Mirrors Dataset.__init__: prompt is the rendered template string,
    # multimodal_inputs comes from the original message list.
    content = [{"type": "text", "text": "Describe this image in detail."}]
    if with_image:
        content.insert(0, {"type": "image", "image": Image.new("RGB", (64, 64), color="red")})
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    multimodal_inputs = process_vision_info(messages, processor) if with_image else None
    return Sample(prompt=prompt, multimodal_inputs=multimodal_inputs)


class TestFilterLongPromptMultimodal:
    def test_multimodal_sample_does_not_crash_and_is_kept(self, tokenizer, processor):
        # Regression: string prompt + images used to crash in qwen_vl_utils
        # (TypeError: string indices must be integers).
        sample = _make_vlm_sample(tokenizer, processor, with_image=True)
        kept = filter_long_prompt([sample], tokenizer, processor, max_length=10_000)
        assert kept == [sample]

    def test_image_token_expansion_counts_toward_limit(self, tokenizer, processor):
        sample = _make_vlm_sample(tokenizer, processor, with_image=True)
        text_len = len(tokenizer(sample.prompt, add_special_tokens=False)["input_ids"])
        full_output = processor(text=sample.prompt, **sample.multimodal_inputs)
        full_len = len(full_output["input_ids"][0])
        assert full_len > text_len

        # A limit between the two lengths must filter the sample; if images
        # were dropped (the old bug), the text-only count would keep it.
        boundary = (text_len + full_len) // 2
        assert filter_long_prompt([sample], tokenizer, processor, max_length=boundary) == []
        assert filter_long_prompt([sample], tokenizer, processor, max_length=full_len) == [sample]

    def test_mixed_batch_filters_each_partition(self, tokenizer, processor):
        text_sample = _make_vlm_sample(tokenizer, processor, with_image=False)
        mm_sample = _make_vlm_sample(tokenizer, processor, with_image=True)
        long_text_sample = Sample(prompt="word " * 500)

        kept = filter_long_prompt(
            [text_sample, mm_sample, long_text_sample], tokenizer, processor, max_length=400
        )
        assert text_sample in kept
        assert mm_sample in kept
        assert long_text_sample not in kept
