import sys
from types import SimpleNamespace

from miles.utils.processing_utils import extract_rollout_video_inputs, process_vision_info


def test_vision_inputs_and_rollout_video_inputs_follow_prompt_order(monkeypatch):
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
                {"type": "video", "video": "first.mp4"},
                {"type": "image", "image": "image.png"},
                {"type": "video", "video": "https://example.test/second.mp4"},
            ],
        }
    ]
    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))

    rollout_video_inputs = extract_rollout_video_inputs(prompt)
    processor_inputs = process_vision_info(prompt, processor)

    assert processor_inputs == {
        "images": ["resolved-image"],
        "videos": ["processed-video-1", "processed-video-2"],
    }
    assert rollout_video_inputs == [
        {"type": "video", "video": "first.mp4"},
        {"type": "video", "video": "https://example.test/second.mp4"},
    ]
    assert calls == {"prompt": prompt, "image_patch_size": 16}


def test_extract_rollout_video_inputs_preserves_the_complete_items():
    video_items = [
        {"type": "video", "video": ["frame-1.png"], "sample_fps": 1},
        {"type": "video", "video": "video.mp4", "fps": 4},
    ]

    assert extract_rollout_video_inputs([{"role": "user", "content": video_items}]) == video_items
