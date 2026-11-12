"""Payload and response handling around SGLang's ``/generate`` endpoint.

    compute_request_payload  ──▶ {"input_ids", "sampling_params", ..., "extra_key"?}
        token space + media  ──▶ + {"mm_token_expansion_start_len": n, "return_prompt_token_ids": True}
        output store         ──▶ + {store request} for a training request with a replay field;
                                 none in evaluation or with the switch off; StorePlanError for media

    adopt_expanded_prompt_ids(sample, payload, output)
        sample.tokens   [prompt ... | response ... | new turn <img> ...]
        engine prompt   placeholders after mm_token_expansion_start_len expanded: <img> ─▶ <img><img><img>
        sample          tokens = engine prompt; tokens added after the first response extend
                        response_length, with loss_mask 0 and rollout_log_probs 0.0

    truncate_to_max_seq_len(args, sample, tokenizer)        only after a request with return_prompt_token_ids
        max_new_tokens was budgeted from the unexpanded prompt, so after expansion the sample can run past
        --max-seq-len: [prompt <img><img><img> | response a b c] ─▶ [prompt <img><img><img> | a], TRUNCATED

The tests pin the extra-key passthrough, when the output store is asked for, that only media requests in token
space ask for expansion, how the engine's expanded prompt replaces the sample's tokens, and that a response past
--max-seq-len is trimmed back to it.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

from miles.rollout.generate_utils import generate_endpoint_utils
from miles.rollout.generate_utils.generate_endpoint_utils import (
    adopt_expanded_prompt_ids,
    compute_request_payload,
    truncate_to_max_seq_len,
    update_sample_from_response,
)
from miles.utils.output_store.plan import OUTPUT_STORE_REQUEST_KEY, StorePlanError
from miles.utils.types import Sample


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_max_response_len=16,
        max_seq_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_rollout_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        rollout_temperature=1.0,
        lora_rank=0,
        lora_adapter_path=None,
        use_output_store=False,
        multimodal_token_space_processing=False,
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


class TestComputeRequestPayloadOutputStore:
    def make_args(self, **overrides) -> SimpleNamespace:
        return SimpleNamespace(**{**vars(_make_args()), "use_output_store": True, **overrides})

    def test_a_training_request_for_replay_asks_for_the_store(self):
        payload, _ = compute_request_payload(
            self.make_args(use_rollout_routing_replay=True), input_ids=[1, 2], sampling_params={}
        )

        assert payload[OUTPUT_STORE_REQUEST_KEY] is True

    @pytest.mark.parametrize(
        ("overrides", "evaluation"),
        [({}, True), ({"use_rollout_routing_replay": False}, False), ({"use_output_store": False}, False)],
        ids=["evaluation", "no-replay-field", "switch-off"],
    )
    def test_no_store_for_evaluation_without_replay_or_without_the_switch(self, overrides, evaluation):
        args = self.make_args(**{"use_rollout_routing_replay": True, **overrides})

        payload, _ = compute_request_payload(args, input_ids=[1, 2], sampling_params={}, evaluation=evaluation)

        assert OUTPUT_STORE_REQUEST_KEY not in payload

    def test_a_multimodal_prompt_is_refused(self, monkeypatch):
        monkeypatch.setattr(generate_endpoint_utils, "encode_image_for_rollout_engine", lambda image: "image")
        with pytest.raises(StorePlanError, match="multimodal"):
            compute_request_payload(
                self.make_args(use_rollout_routing_replay=True),
                input_ids=[1, 2],
                sampling_params={},
                multimodal_inputs={"images": [object()]},
            )


class TestTokenSpaceRequest:
    def test_media_requests_ask_for_expansion_after_the_prefix(self, tmp_path):
        args = _make_args()
        args.multimodal_token_space_processing = True
        multimodal_inputs = {"images": [str(tmp_path / "a.png"), str(tmp_path / "b.png")]}

        payload, _ = compute_request_payload(
            args,
            input_ids=[1, 2, 3],
            sampling_params={},
            multimodal_inputs=multimodal_inputs,
            mm_token_expansion_start_len=2,
        )

        assert payload["image_data"] == multimodal_inputs["images"]
        assert payload["mm_token_expansion_start_len"] == 2
        assert payload["return_prompt_token_ids"] is True

    def test_text_only_requests_carry_no_token_space_fields(self):
        args = _make_args()
        args.multimodal_token_space_processing = True

        payload, _ = compute_request_payload(args, input_ids=[1, 2], sampling_params={})

        assert "mm_token_expansion_start_len" not in payload
        assert "return_prompt_token_ids" not in payload


class TestAdoptExpandedPromptIds:
    def test_first_turn_takes_the_engine_prompt(self):
        sample = Sample(tokens=[1, 9, 2])

        adopt_expanded_prompt_ids(sample, {"return_prompt_token_ids": True}, {"prompt_token_ids": [1, 9, 9, 9, 2]})

        assert sample.tokens == [1, 9, 9, 9, 2]
        assert sample.response_length == 0

    def test_later_expansion_is_a_zero_loss_suffix(self):
        # [prompt 1 9 | response 5 | new turn 9 2]; the engine expands the new turn's 9 into three tokens.
        sample = Sample(tokens=[1, 9, 5, 9, 2], response_length=3, loss_mask=[1, 0, 0], rollout_log_probs=[-0.1, 0, 0])

        adopt_expanded_prompt_ids(
            sample, {"return_prompt_token_ids": True}, {"prompt_token_ids": [1, 9, 5, 9, 9, 9, 2]}
        )

        assert sample.tokens == [1, 9, 5, 9, 9, 9, 2]
        assert sample.response_length == 5
        assert sample.loss_mask == [1, 0, 0, 0, 0]
        assert sample.rollout_log_probs == [-0.1, 0, 0, 0.0, 0.0]

    def test_requests_without_the_flag_keep_their_tokens(self):
        sample = Sample(tokens=[1, 2])

        adopt_expanded_prompt_ids(sample, {}, {"prompt_token_ids": [7]})

        assert sample.tokens == [1, 2]


class _Tokenizer:
    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


class TestTruncateToMaxSeqLen:
    def test_a_response_past_the_limit_is_cut_back_to_it(self):
        # [prompt 1 9 9 9 | response 5 6 7] with --max-seq-len 5: the last two output tokens go.
        sample = Sample(
            tokens=[1, 9, 9, 9, 5, 6, 7], response_length=3, loss_mask=[1, 1, 1], rollout_log_probs=[-0.1, -0.2, -0.3]
        )

        truncate_to_max_seq_len(SimpleNamespace(max_seq_len=5), sample, _Tokenizer())

        assert sample.tokens == [1, 9, 9, 9, 5]
        assert sample.response_length == 1
        assert sample.loss_mask == [1]
        assert sample.rollout_log_probs == [-0.1]
        assert sample.response == "5"
        assert sample.status == Sample.Status.TRUNCATED

    def test_a_prompt_past_the_limit_loses_its_whole_response(self):
        # [prompt 1 9 9 9 | response 5] with --max-seq-len 3: the prompt alone is over the limit, so the sample ends
        # truncated with no response, as it would had the prompt left no budget before the request.
        sample = Sample(tokens=[1, 9, 9, 9, 5], response_length=1, loss_mask=[1], rollout_log_probs=[-0.1])

        truncate_to_max_seq_len(SimpleNamespace(max_seq_len=3), sample, _Tokenizer())

        assert sample.tokens == [1, 9, 9, 9]
        assert sample.response_length == 0
        assert sample.loss_mask == []
        assert sample.response == ""
        assert sample.status == Sample.Status.TRUNCATED

    @pytest.mark.parametrize("max_seq_len", [None, 5, 8])
    def test_a_sample_within_the_limit_is_left_alone(self, max_seq_len):
        sample = Sample(tokens=[1, 2, 3, 4, 5], response_length=2, status=Sample.Status.COMPLETED)

        truncate_to_max_seq_len(SimpleNamespace(max_seq_len=max_seq_len), sample, _Tokenizer())

        assert sample.tokens == [1, 2, 3, 4, 5]
        assert sample.response_length == 2
        assert sample.status == Sample.Status.COMPLETED

    def test_update_sample_from_response_trims_an_expanded_turn(self):
        """The engine expands <img> (9) into three tokens after max_new_tokens was budgeted from [1, 9, 2]."""
        args = SimpleNamespace(**vars(_make_args()), sglang=SimpleNamespace(common_value=lambda key: None))
        args.max_seq_len = 6
        payload = {"input_ids": [1, 9, 2], "return_prompt_token_ids": True, "sampling_params": {"max_new_tokens": 3}}
        output = {
            "text": "5 6 7",
            "prompt_token_ids": [1, 9, 9, 9, 2],
            "meta_info": {
                "output_token_logprobs": [(-0.1, 5), (-0.2, 6), (-0.3, 7)],
                "finish_reason": {"type": "stop"},
            },
        }
        sample = Sample()

        asyncio.run(update_sample_from_response(args, sample, payload, output, tokenizer=_Tokenizer()))

        assert sample.tokens == [1, 9, 9, 9, 2, 5]
        assert sample.response_length == 1
        assert sample.rollout_log_probs == [-0.1]
        assert sample.response == "5"
        assert sample.status == Sample.Status.TRUNCATED

    def test_update_sample_from_response_leaves_a_legacy_request_alone(self):
        """Without the token-space flag the prompt was expanded before the budget, so nothing is ever trimmed."""
        args = SimpleNamespace(**vars(_make_args()), sglang=SimpleNamespace(common_value=lambda key: None))
        args.max_seq_len = 4
        payload = {"input_ids": [1, 9, 2], "sampling_params": {"max_new_tokens": 3}}
        output = {
            "text": "5 6 7",
            "meta_info": {
                "output_token_logprobs": [(-0.1, 5), (-0.2, 6), (-0.3, 7)],
                "finish_reason": {"type": "stop"},
            },
        }
        sample = Sample()

        asyncio.run(update_sample_from_response(args, sample, payload, output, tokenizer=_Tokenizer()))

        assert sample.tokens == [1, 9, 2, 5, 6, 7]
        assert sample.response == "5 6 7"
        assert sample.status == Sample.Status.COMPLETED


def test_routed_experts_cover_the_prompt_the_engine_served():
    layers, topk, response = 2, 3, [7, 8]
    sample = Sample(prompt="", tokens=[1, -101, 2])
    served_prompt_tokens = 6  # the placeholder became a 4-token image run
    rows = np.arange((served_prompt_tokens + len(response) - 1) * layers * topk, dtype=np.int32)
    output = {
        "text": "",
        "meta_info": {
            "output_token_logprobs": [(-0.5, t, None) for t in response],
            "finish_reason": {"type": "stop"},
            "prompt_tokens": served_prompt_tokens,
            "completion_tokens": len(response),
            "routed_experts": pybase64.b64encode(rows.tobytes()).decode(),
        },
    }
    args = SimpleNamespace(
        rollout_top_logprobs_num=0, sglang=make_sglang_config(speculative_algorithm=None), num_layers=layers
    )

    # A request without return_prompt_token_ids never trims, so a custom generate function may omit the tokenizer.
    asyncio.run(update_sample_from_response(args, sample, {"input_ids": sample.tokens}, output))

    assert sample.rollout_routed_experts.shape == (served_prompt_tokens + len(response) - 1, layers, topk)
