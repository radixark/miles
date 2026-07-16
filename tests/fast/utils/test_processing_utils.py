import sys
from types import SimpleNamespace

import pytest

from miles.utils.processing_utils import prepare_rollout_video_sources, process_vision_info


def test_video_config_is_shared_by_local_processing_and_rollout(monkeypatch):
    calls = {}

    def fake_process_vision_info(prompt, image_patch_size):
        calls["prompt"] = prompt
        calls["image_patch_size"] = image_patch_size
        return ["resolved-image"], ["processed-video-1", "processed-video-2"]

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
    }
    assert rollout_video_sources == ["first.mp4", "https://example.test/second.mp4"]
    assert [item["fps"] for item in prompt[0]["content"] if item["type"] == "video"] == [4, 4]
    assert calls == {"prompt": prompt, "image_patch_size": 16}


@pytest.mark.parametrize("video_item", [{"fps": 2}, {"video_start": 1}])
def test_per_video_options_must_match_the_sglang_config(video_item):
    prompt = [{"role": "user", "content": [{"type": "video", "video": "video.mp4", **video_item}]}]
    with pytest.raises(NotImplementedError):
        prepare_rollout_video_sources(prompt, {"fps": 4})
