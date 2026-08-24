from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import sys
from types import SimpleNamespace

import pytest

from miles.utils.processing_utils import (
    build_processor_kwargs,
    extract_multimodal_train_inputs,
    prepare_rollout_video_sources,
    process_vision_info,
)


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


def test_video_config_is_shared_by_local_processing_and_rollout(monkeypatch):
    calls = {}

    def fake_process_vision_info(prompt, image_patch_size, return_video_kwargs, return_video_metadata):
        calls["prompt"] = prompt
        calls["image_patch_size"] = image_patch_size
        calls["return_video_kwargs"] = return_video_kwargs
        calls["return_video_metadata"] = return_video_metadata
        videos = [("processed-video-1", {"fps": 1.0}), ("processed-video-2", {"fps": 2.0})]
        return ["resolved-image"], videos, {"do_sample_frames": False}

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=fake_process_vision_info),
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "first.mp4", "fps": 4},
                {"type": "image", "image": "image.png"},
                {"type": "video", "video": "https://example.test/second.mp4"},
            ],
        }
    ]
    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))

    rollout_video_sources = prepare_rollout_video_sources(prompt, {"fps": 4})
    processor_inputs = process_vision_info(prompt, processor)

    assert processor_inputs == {
        "images": ["resolved-image"],
        "videos": ["processed-video-1", "processed-video-2"],
        "video_metadata": [{"fps": 1.0}, {"fps": 2.0}],
    }
    assert rollout_video_sources == ["first.mp4", "https://example.test/second.mp4"]
    assert [item["fps"] for item in prompt[0]["content"] if item["type"] == "video"] == [4, 4]
    assert calls == {
        "prompt": prompt,
        "image_patch_size": 16,
        "return_video_kwargs": True,
        "return_video_metadata": True,
    }


def test_process_vision_info_keeps_image_only_contract(monkeypatch):
    def fake_process_vision_info(prompt, image_patch_size, return_video_kwargs, return_video_metadata):
        return ["resolved-image"], None, {"do_sample_frames": False}

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=fake_process_vision_info),
    )
    prompt = [{"role": "user", "content": [{"type": "image", "image": "image.png"}]}]
    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))

    assert process_vision_info(prompt, processor) == {"images": ["resolved-image"], "videos": None}


def test_video_metadata_routes_into_videos_kwargs():
    kwargs = build_processor_kwargs(
        {
            "videos": ["decoded-video"],
            "video_metadata": [{"fps": 1.0, "frames_indices": [0, 1]}],
        }
    )

    # The metadata must reach the processor as videos_kwargs (matching the
    # rollout engine's server-side call) and never as a top-level kwarg.
    assert "video_metadata" not in kwargs
    assert kwargs["videos_kwargs"]["video_metadata"] == [{"fps": 1.0, "frames_indices": [0, 1]}]
    assert kwargs["videos_kwargs"]["do_sample_frames"] is False
    assert kwargs["videos_kwargs"]["return_tensors"] == "pt"
    assert kwargs["videos"] == ["decoded-video"]


@pytest.mark.parametrize("video_item", [{"fps": 2}, {"video_start": 1}])
def test_per_video_options_must_match_the_sglang_config(video_item):
    prompt = [{"role": "user", "content": [{"type": "video", "video": "video.mp4", **video_item}]}]
    with pytest.raises(NotImplementedError):
        prepare_rollout_video_sources(prompt, {"fps": 4})


@pytest.mark.parametrize("unsupported_option", ["nframes", "min_pixels", "max_pixels", "total_pixels"])
def test_video_config_rejects_options_forwarded_illegally_by_sglang(unsupported_option):
    prompt = [{"role": "user", "content": [{"type": "video", "video": "video.mp4"}]}]

    with pytest.raises(ValueError, match=unsupported_option):
        prepare_rollout_video_sources(prompt, {unsupported_option: 4})


@pytest.mark.parametrize(
    "prompt,expected_error",
    [
        (["not-a-message"], ValueError),
        ([{"role": "user", "content": ["not-an-item"]}], ValueError),
        (
            [{"role": "user", "content": [{"type": "video", "video": ["frame-1.png", "frame-2.png"]}]}],
            NotImplementedError,
        ),
    ],
)
def test_rollout_video_sources_reject_malformed_or_in_memory_inputs(prompt, expected_error):
    with pytest.raises(expected_error):
        prepare_rollout_video_sources(prompt, {"fps": 2})
