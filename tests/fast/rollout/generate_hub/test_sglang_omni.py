import asyncio
import base64
from types import SimpleNamespace

import pytest
import torch

from miles.rollout.generate_utils.generate_endpoint_utils import serialize_multimodal_train_inputs
from miles.utils.types import Sample


def test_serialize_audio_video_processor_tensors():
    inputs = {
        "input_features": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        "feature_attention_mask": torch.ones((1, 2), dtype=torch.long),
        "pixel_values_videos": torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3),
        "video_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "video_second_per_grid": torch.tensor([0.5], dtype=torch.float32),
    }

    bundle = serialize_multimodal_train_inputs(inputs)

    assert bundle["version"] == 1
    assert bundle["modalities"] == ["audio", "video"]
    assert set(bundle["tensors"]) == set(inputs)
    for name, tensor in inputs.items():
        encoded = bundle["tensors"][name]
        assert encoded["dtype"] == str(tensor.dtype).removeprefix("torch.")
        assert encoded["shape"] == list(tensor.shape)
        assert base64.b64decode(encoded["data"]) == (
            tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )


def test_qwen_omni_media_extraction_and_tensor_normalization(monkeypatch):
    from miles.utils import processing_utils

    qwen_omni_utils = pytest.importorskip("qwen_omni_utils")
    prompt = [{"role": "user", "content": [{"type": "audio", "audio": "a.wav"}]}]
    captured = {}

    def fake_process_mm_info(conversations, **kwargs):
        captured["conversations"], captured["kwargs"] = conversations, kwargs
        return ["audio samples"], ["image"], ["video frames"]

    monkeypatch.setattr(qwen_omni_utils, "process_mm_info", fake_process_mm_info)
    processor = SimpleNamespace(audio_token="<audio>", image_processor=SimpleNamespace(patch_size=16))

    media = processing_utils.process_vision_info(prompt, processor)
    train_inputs = processing_utils.extract_multimodal_train_inputs(
        {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "pixel_values_videos": torch.ones((1, 2)),
            "video_second_per_grid": [0.5],
        }
    )

    assert media == {
        "audio": ["audio samples"],
        "images": ["image"],
        "videos": ["video frames"],
    }
    assert captured == {
        "conversations": prompt,
        "kwargs": {"use_audio_in_video": False, "image_patch_size": 16},
    }
    assert torch.equal(train_inputs["video_second_per_grid"], torch.tensor([0.5]))


def test_sglang_omni_adapter_sends_processed_audio_video(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    captured = {}

    async def fake_post(url, payload, headers=None):
        captured.update(url=url, payload=payload, headers=headers)
        return {}

    async def fake_update(*args, **kwargs):
        return None

    monkeypatch.setattr(sglang_omni, "post", fake_post)
    monkeypatch.setattr(sglang_omni, "update_sample_from_response", fake_update)
    monkeypatch.setattr(
        sglang_omni,
        "compute_prompt_ids_from_sample",
        lambda state, sample: [1, 2, 3],
    )

    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_policy="round_robin",
        rollout_max_response_len=128,
        rollout_max_context_len=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
    )
    sample = Sample(
        multimodal_inputs={"audio": ["audio"], "videos": ["video"]},
        multimodal_train_inputs={
            "input_features": torch.ones((1, 2, 3)),
            "pixel_values_videos": torch.ones((2, 2, 3)),
        },
    )
    generate_input = SimpleNamespace(
        args=args,
        sample=sample,
        sampling_params={"temperature": 1.0, "max_new_tokens": 16},
        state=None,
    )

    asyncio.run(sglang_omni.generate(generate_input))

    assert captured["url"].endswith("/generate")
    assert captured["payload"]["input_ids"] == [1, 2, 3]
    assert captured["payload"]["multimodal_train_inputs"]["modalities"] == [
        "audio",
        "video",
    ]
    assert captured["headers"] == {"x-sglang-omni-route-capabilities": "audio_input,video_input"}
