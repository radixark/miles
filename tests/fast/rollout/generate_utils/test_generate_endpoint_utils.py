"""Payload and response handling around SGLang's ``/generate`` endpoint.

    compute_request_payload  ──▶ {"input_ids", "sampling_params", ..., "extra_key"?}
        token space + media  ──▶ + {"mm_token_expansion_start_len": n, "return_prompt_token_ids": True}

    adopt_expanded_prompt_ids(sample, payload, output)
        sample.tokens   [prompt ... | response ... | new turn <img> ...]
        engine prompt   placeholders after mm_token_expansion_start_len expanded: <img> ─▶ <img><img><img>
        sample          tokens = engine prompt; tokens added after the first response extend
                        response_length, with loss_mask 0 and rollout_log_probs 0.0

The tests pin the extra-key passthrough, that only media requests in token space ask for expansion, and how the
engine's expanded prompt replaces the sample's tokens.
"""

from types import SimpleNamespace

from miles.rollout.generate_utils.generate_endpoint_utils import adopt_expanded_prompt_ids, compute_request_payload
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
