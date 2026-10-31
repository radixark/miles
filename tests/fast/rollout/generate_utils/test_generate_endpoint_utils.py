"""One image placeholder per image on the wire; the expanded prompt in the sample."""

import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils import generate_endpoint_utils
from miles.rollout.generate_utils.generate_endpoint_utils import (
    collapse_image_pads,
    compute_request_payload,
    image_pad_id_for,
    update_sample_from_response,
)
from miles.utils.output_store.plan import OUTPUT_STORE_REQUEST_KEY, StorePlanError
from miles.utils.types import Sample

PAD = 151655


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        rollout_temperature=1.0,
        lora_rank=0,
        lora_adapter_path=None,
        use_output_store=False,
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
                input_ids=[1, PAD, 2],
                sampling_params={},
                multimodal_inputs={"images": [object()]},
                image_pad_id=PAD,
            )


def _args(**over):
    base = dict(
        rollout_max_response_len=64,
        rollout_max_context_len=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        rollout_top_logprobs_num=0,
        sampling_logprobs_mode="selected",
        rollout_temperature=1.0,
        lora_rank=0,
        lora_adapter_path=None,
        use_sampling_support_replay=False,
        sglang=SimpleNamespace(common_value=lambda name: None),
        rollout_sampling_mask=False,
        use_output_store=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_collapse_keeps_one_pad_per_run():
    ids = [1, 2, 151652, PAD, PAD, PAD, 151653, 3, 151652, PAD, PAD, 151653, 4]
    assert collapse_image_pads(ids, PAD) == [1, 2, 151652, PAD, 151653, 3, 151652, PAD, 151653, 4]
    assert collapse_image_pads([1, 2, 3], PAD) == [1, 2, 3]
    assert collapse_image_pads([], PAD) == []


def test_image_pad_id_comes_from_the_family_table_then_the_processor():
    from miles.utils.chat_template_utils.tito_tokenizer import ImageTokens

    tok = SimpleNamespace(
        convert_tokens_to_ids=lambda t: {"<|image_pad|>": PAD, "<|image|>": 7}.get(t, 0), unk_token_id=0
    )
    processor = SimpleNamespace(image_token="<|image_pad|>")
    # the TITO family's table entry wins over the processor's token
    glm = SimpleNamespace(
        image_tokens=ImageTokens("<|begin_of_image|>", "<|image|>", "<|end_of_image|>"),
        image_token_ids=lambda: {"begin": 5, "placeholder": 7, "end": 6},
    )
    assert image_pad_id_for(SimpleNamespace(tokenizer=tok, processor=processor, tito_tokenizer=glm)) == 7
    # a family whose placeholder the engine does not expand collapses nothing
    fixed = SimpleNamespace(
        image_tokens=ImageTokens(None, "<|image|>", None, expands=False), image_token_ids=lambda: {"placeholder": 7}
    )
    assert image_pad_id_for(SimpleNamespace(tokenizer=tok, processor=processor, tito_tokenizer=fixed)) is None
    # a family without an entry falls through to the processor's token
    text_family = SimpleNamespace(image_tokens=None, image_token_ids=lambda: None)
    assert image_pad_id_for(SimpleNamespace(tokenizer=tok, processor=processor, tito_tokenizer=text_family)) == PAD
    assert image_pad_id_for(SimpleNamespace(tokenizer=tok, processor=processor)) == PAD
    assert image_pad_id_for(SimpleNamespace(tokenizer=tok, processor=None)) is None
    unknown = SimpleNamespace(tokenizer=tok, processor=SimpleNamespace(image_token="<|no_such|>"))
    assert image_pad_id_for(unknown) is None, "an unknown token must not collapse anything"


def test_images_without_a_known_placeholder_are_refused(monkeypatch):
    import pytest

    import miles.rollout.generate_utils.generate_endpoint_utils as m

    monkeypatch.setattr(m, "encode_image_for_rollout_engine", lambda image: f"b64:{image}")
    with pytest.raises(ValueError, match="image-token table"):
        compute_request_payload(_args(), [1, PAD, 2], {}, multimodal_inputs={"images": ["img"]}, image_pad_id=None)


def test_request_carries_one_pad_per_image_and_counts_the_expanded_length(monkeypatch):
    import miles.rollout.generate_utils.generate_endpoint_utils as m

    monkeypatch.setattr(m, "encode_image_for_rollout_engine", lambda image: f"b64:{image}")
    expanded = [1, 151652] + [PAD] * 5 + [151653, 2]
    payload, status = compute_request_payload(
        _args(rollout_max_context_len=12),
        expanded,
        {"temperature": 1.0},
        multimodal_inputs={"images": ["img0"]},
        image_pad_id=PAD,
    )
    assert status is None
    assert payload["input_ids"] == [1, 151652, PAD, 151653, 2]
    assert payload["image_data"] == ["b64:img0"]
    # the context arithmetic uses the expanded length the engine will run
    assert payload["sampling_params"]["max_new_tokens"] == 12 - len(expanded)
    # no images: nothing to collapse even with a pad id
    payload, _ = compute_request_payload(_args(), [1, PAD, PAD, 2], {}, image_pad_id=PAD)
    assert payload["input_ids"] == [1, PAD, PAD, 2]


def test_sample_keeps_the_expanded_prompt_ids():
    expanded = [1, 151652] + [PAD] * 5 + [151653, 2]
    payload = {"input_ids": collapse_image_pads(expanded, PAD)}
    output = {
        "text": "hi",
        "meta_info": {"output_token_logprobs": [(-0.1, 7), (-0.2, 8)], "finish_reason": {"type": "stop"}},
    }
    sample = Sample()
    asyncio.run(update_sample_from_response(_args(), sample, payload=payload, output=output, prompt_ids=expanded))
    assert sample.tokens == expanded + [7, 8]
    assert sample.response_length == 2
    # default: the request's ids, as before
    sample = Sample()
    asyncio.run(update_sample_from_response(_args(), sample, payload=payload, output=output))
    assert sample.tokens == payload["input_ids"] + [7, 8]
