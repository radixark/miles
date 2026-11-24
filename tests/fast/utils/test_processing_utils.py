import sys
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from miles.utils.processing_utils import extract_multimodal_train_inputs, process_vision_info


def test_extract_multimodal_train_inputs_drops_qwen3_vl_token_metadata():
    pixel_values = object()
    image_grid_thw = object()
    processor_output = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "mm_token_type_ids": [[0, 1, 0]],
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    assert extract_multimodal_train_inputs(processor_output) == {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    assert (
        extract_multimodal_train_inputs(
            {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "mm_token_type_ids": [[0, 1, 0]],
            }
        )
        is None
    )


_TEXT_PROMPTS = [
    [{"role": "user", "content": "Hello"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    [[{"role": "user", "content": [{"type": "text", "text": "Batched"}]}]],
]


@pytest.mark.parametrize("prompt", _TEXT_PROMPTS)
def test_text_only_prompts_need_no_vision_dependencies(monkeypatch, prompt):
    """A text-only checkpoint can still ship a processor; its prompts must not import qwen_vl_utils
    or read an image processor it does not have."""
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)  # makes the import raise ImportError
    assert process_vision_info(prompt, SimpleNamespace()) == {"images": None, "videos": None}


@pytest.mark.parametrize(
    "item",
    [
        {"type": "image", "image": "a.png"},
        {"type": "image_url", "image_url": {"url": "a.png"}},
        {"image": "a.png"},
        {"type": "video", "video": "a.mp4"},
    ],
)
def test_vision_prompts_still_go_through_qwen_vl_utils(monkeypatch, item):
    calls = []

    def fake_process_vision_info(prompt, image_patch_size):
        calls.append((prompt, image_patch_size))
        return ["image"], None

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=fake_process_vision_info))
    prompt = [{"role": "user", "content": [item, {"type": "text", "text": "What is this?"}]}]
    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))
    assert process_vision_info(prompt, processor) == {"images": ["image"], "videos": None}
    assert calls == [(prompt, 16)]


def test_vision_prompts_without_qwen_vl_utils_fail_loudly(monkeypatch):
    """Images must not be dropped silently when the dependency is missing."""
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)
    prompt = [{"role": "user", "content": [{"type": "image", "image": "a.png"}]}]
    with pytest.raises(ImportError):
        process_vision_info(prompt, SimpleNamespace(image_processor=SimpleNamespace()))
