from types import SimpleNamespace

import pytest
from PIL import Image

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import single_turn
from miles.rollout.generate_utils.generate_endpoint_utils import (
    build_rollout_media_payload,
    compute_prompt_ids_from_sample,
    compute_request_payload,
)
from miles.utils.types import Sample

PROCESSOR_PROMPT_IDS = [100, 101, 102]
ROLLOUT_PROMPT_IDS = [1, 2]


class _Processor:
    def __init__(self):
        self.text = None

    def __call__(self, text, **kwargs):
        self.text = text
        return {"input_ids": [PROCESSOR_PROMPT_IDS], "pixel_values_videos": "train-only"}


class _Tokenizer:
    def apply_chat_template(self, prompt, **kwargs):
        return "<video>rendered prompt"

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return ROLLOUT_PROMPT_IDS


def _args(**overrides):
    defaults = dict(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_path=None,
        sglang_speculative_algorithm=None,
    )
    return SimpleNamespace(**(defaults | overrides))


def _video_sample():
    return Sample(
        prompt="<video>describe it",
        multimodal_inputs={"videos": [object()]},
        rollout_video_sources=["video.mp4"],
    )


def test_multimodal_request_contract():
    image = Image.new("RGB", (2, 2), color="red")
    media_payload = build_rollout_media_payload(
        {"images": [image], "videos": [object()]},
        ["https://example.test/video.mp4"],
    )
    request, status = compute_request_payload(
        _args(rollout_max_context_len=5),
        input_ids=PROCESSOR_PROMPT_IDS,
        rollout_input_ids=ROLLOUT_PROMPT_IDS,
        sampling_params={"max_new_tokens": 10},
    )

    assert media_payload["image_data"][0].startswith("data:image/png;base64,")
    assert media_payload["video_data"] == ["https://example.test/video.mp4"]
    assert status is None
    assert request["input_ids"] == ROLLOUT_PROMPT_IDS
    assert request["sampling_params"]["max_new_tokens"] == 2


def test_prompt_processing_keeps_training_and_rollout_ids_separate():
    sample = _video_sample()
    sample.prompt = [{"role": "user", "content": [{"type": "video", "video": "video.mp4"}]}]
    processor = _Processor()
    state = SimpleNamespace(processor=processor, tokenizer=_Tokenizer())

    prompt_ids = compute_prompt_ids_from_sample(state, sample)

    assert prompt_ids == PROCESSOR_PROMPT_IDS
    assert sample.rollout_prompt_ids == ROLLOUT_PROMPT_IDS
    assert sample.multimodal_train_inputs == {"pixel_values_videos": "train-only"}
    assert processor.text == "<video>rendered prompt"


def test_image_only_keeps_the_existing_request_contract():
    image = Image.new("RGB", (2, 2), color="red")
    sample = Sample(prompt="<image>describe it", multimodal_inputs={"images": [image]})
    state = SimpleNamespace(processor=_Processor(), tokenizer=_Tokenizer())
    prompt_ids = compute_prompt_ids_from_sample(state, sample)

    payload, _ = compute_request_payload(
        _args(), prompt_ids, {"max_new_tokens": 4}, multimodal_inputs=sample.multimodal_inputs
    )

    assert sample.rollout_prompt_ids is None
    assert payload["input_ids"] == PROCESSOR_PROMPT_IDS
    assert payload["image_data"][0].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_single_turn_sends_rollout_ids_but_keeps_processor_ids(monkeypatch):
    requests = []

    async def fake_post(url, payload):
        requests.append(payload)
        return {
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [(-0.1, 20)],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(single_turn, "post", fake_post)
    args = _args()
    sample = _video_sample()
    state = SimpleNamespace(args=args, processor=_Processor(), tokenizer=_Tokenizer())

    output = await single_turn.generate(
        GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 4}, evaluation=False)
    )

    assert requests[0]["input_ids"] == ROLLOUT_PROMPT_IDS
    assert requests[0]["video_data"] == ["video.mp4"]
    assert output.samples.tokens == PROCESSOR_PROMPT_IDS + [20]
