import asyncio
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.utils.types import Sample


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        lora_rank=0,
        lora_adapter_path=None,
    )


class TestComputeRequestPayloadExtraKey:
    def test_no_extra_key_leaves_the_payload_without_one(self):
        """Without an extra key the payload has no extra_key field at all."""
        payload, halt_status = compute_request_payload(_make_args(), input_ids=[1, 2], sampling_params={})

        assert halt_status is None
        assert "extra_key" not in payload

    def test_an_extra_key_reaches_the_payload_verbatim(self):
        """The extra key is forwarded unchanged to the engine request."""
        payload, halt_status = compute_request_payload(
            _make_args(), input_ids=[1, 2], sampling_params={}, kv_cache_namespace="train:-:7"
        )

        assert halt_status is None
        assert payload["extra_key"] == "train:-:7"


_NUM_LAYERS, _TOPK = 2, 4


def _response(prompt_tokens: int, response_tokens: list[int], routed_rows: int) -> tuple[dict, np.ndarray]:
    routed = np.arange(routed_rows * _NUM_LAYERS * _TOPK, dtype=np.int32).reshape(routed_rows, _NUM_LAYERS, _TOPK)
    meta_info = {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": prompt_tokens,
        "output_token_logprobs": [(-0.5, token) for token in response_tokens],
        "routed_experts": pybase64.b64encode(routed.tobytes()).decode("ascii"),
    }
    return {"text": "x", "meta_info": meta_info}, routed


def _update(payload_ids: list[int], output: dict) -> Sample:
    args = SimpleNamespace(num_layers=_NUM_LAYERS, sglang_speculative_algorithm=None)
    sample = Sample(prompt="p")
    asyncio.run(update_sample_from_response(args, sample, payload={"input_ids": payload_ids}, output=output))
    return sample


def test_routed_experts_cover_the_prompt_the_engine_expanded():
    # One image placeholder the engine expanded into 171 tokens: 3 request tokens become 173.
    output, routed = _response(prompt_tokens=173, response_tokens=[11, 12], routed_rows=173 + 2 - 1)

    sample = _update([5, 128001, 8], output)

    assert sample.tokens == [5, 128001, 8, 11, 12]
    np.testing.assert_array_equal(sample.rollout_routed_experts, routed)


def test_text_prompts_keep_one_routing_row_per_token_but_the_last():
    output, routed = _response(prompt_tokens=3, response_tokens=[11, 12], routed_rows=3 + 2 - 1)

    sample = _update([5, 6, 8], output)

    assert sample.rollout_routed_experts.shape == (len(sample.tokens) - 1, _NUM_LAYERS, _TOPK)
    np.testing.assert_array_equal(sample.rollout_routed_experts, routed)


def _max_new_tokens(*, evaluation: bool, rollout_max_context_len: int | None, eval_max_context_len: int | None) -> int:
    args = _make_args()
    args.rollout_max_response_len = 100
    args.rollout_max_context_len = rollout_max_context_len
    args.eval_max_context_len = eval_max_context_len
    payload, halt_status = compute_request_payload(
        args, input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 50}, evaluation=evaluation
    )
    assert halt_status is None
    return payload["sampling_params"]["max_new_tokens"]


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        # A trainer-imposed window caps rollouts ...
        (False, 10 - 3),
        # ... but eval samples never reach the trainer and keep their own, looser cap.
        (True, 40 - 3),
    ],
)
def test_rollout_and_eval_requests_each_honor_their_own_context_cap(evaluation, expected):
    assert _max_new_tokens(evaluation=evaluation, rollout_max_context_len=10, eval_max_context_len=40) == expected
