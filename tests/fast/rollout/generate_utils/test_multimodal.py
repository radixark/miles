from types import SimpleNamespace

import pytest

from examples.experimental.video_rollout import generate as video_rollout
from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample

PROCESSOR_PROMPT_IDS = [100, 101, 102]
ROLLOUT_PROMPT_IDS = [1, 2]


class _Processor:
    def __call__(self, text, **kwargs):
        return {"input_ids": [PROCESSOR_PROMPT_IDS], "pixel_values_videos": "train-only"}


class _Tokenizer:
    def encode(self, text, add_special_tokens):
        return ROLLOUT_PROMPT_IDS


def _args(**overrides):
    defaults = dict(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        partial_rollout=True,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_path=None,
        sglang_speculative_algorithm=None,
    )
    return SimpleNamespace(**(defaults | overrides))


def _sample(**overrides):
    values = dict(
        prompt="<video>describe it",
        multimodal_inputs={"videos": [object()]},
        rollout_video_sources=["video.mp4"],
    )
    return Sample(**(values | overrides))


def _state(args):
    return SimpleNamespace(args=args, processor=_Processor(), tokenizer=_Tokenizer())


def _response(token_id, finish_reason="stop"):
    return {
        "text": "answer",
        "meta_info": {
            "output_token_logprobs": [(-0.1, token_id)],
            "finish_reason": {"type": finish_reason},
        },
    }


@pytest.mark.asyncio
async def test_single_turn_uses_video_prompt_and_resumes_partial_output(monkeypatch):
    requests = []

    async def fake_post(url, payload):
        requests.append(payload)
        return _response(21)

    monkeypatch.setattr(video_rollout, "post", fake_post)
    args = _args()
    sample = _sample(
        tokens=PROCESSOR_PROMPT_IDS + [20],
        response="partial",
        response_length=1,
        loss_mask=[0],
        status=Sample.Status.ABORTED,
    )

    output = await video_rollout.single_turn(
        GenerateFnInput(_state(args), sample, {"max_new_tokens": 4}, False)
    )

    assert requests[0]["input_ids"] == ROLLOUT_PROMPT_IDS + [20]
    assert requests[0]["video_data"] == ["video.mp4"]
    assert requests[0]["sampling_params"]["max_new_tokens"] == 3
    assert output.samples.tokens == PROCESSOR_PROMPT_IDS + [20, 21]
    assert output.samples.loss_mask == [0, 1]
    assert output.samples.multimodal_train_inputs == {"pixel_values_videos": "train-only"}


@pytest.mark.asyncio
async def test_multi_turn_replays_video_with_each_context_suffix(monkeypatch):
    requests = []

    async def fake_post(url, payload):
        requests.append(payload)
        return _response(20 if len(requests) == 1 else 21)

    class _Parser:
        calls = 0

        def parse_non_stream(self, text):
            self.calls += 1
            return None, [object()] if self.calls == 1 else []

    async def fake_execute_tool_calls(tool_calls, execute_one):
        return [{"role": "tool", "content": "result"}]

    def fake_update_with_tool_response(sample, tool_messages, tokenizer):
        sample.tokens.append(30)
        sample.response_length += 1
        sample.loss_mask.append(0)
        sample.rollout_log_probs.append(0.0)

    monkeypatch.setattr(video_rollout, "post", fake_post)
    monkeypatch.setattr(video_rollout, "load_function", lambda path: [] if path == "tools" else None)
    monkeypatch.setattr(video_rollout, "create_tool_call_parser", lambda *args: _Parser())
    monkeypatch.setattr(video_rollout, "execute_tool_calls", fake_execute_tool_calls)
    monkeypatch.setattr(video_rollout, "update_sample_with_tool_responses", fake_update_with_tool_response)
    args = _args(
        partial_rollout=False,
        generate_execute_tool_function_path="execute",
        generate_tool_specs_path="tools",
        generate_tool_call_parser="parser",
        generate_max_turns=2,
        generate_multi_samples=False,
    )

    output = await video_rollout.multi_turn(
        GenerateFnInput(_state(args), _sample(), {"max_new_tokens": 4}, False)
    )

    assert [request["input_ids"] for request in requests] == [
        ROLLOUT_PROMPT_IDS,
        ROLLOUT_PROMPT_IDS + [20, 30],
    ]
    assert all(request["video_data"] == ["video.mp4"] for request in requests)
    assert output.samples.tokens == PROCESSOR_PROMPT_IDS + [20, 30, 21]
