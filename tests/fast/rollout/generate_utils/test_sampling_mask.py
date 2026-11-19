import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.score_centering_fixtures import _flat_sampling_mask
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.rollout.generate_utils.sampling_mask import (
    append_forced_sampling_tokens,
    append_sampling_metadata,
    configure_sampling_mask_request,
    merge_sampling_masks,
    read_sampled_logprobs,
    read_selected_sampling_logprobs,
    should_return_sampling_mask,
)
from miles.utils.output_store.plan import OUTPUT_STORE_REQUEST_KEY, StorePlanError
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


def _payload_args(**overrides):
    return SimpleNamespace(
        **{
            "use_rollout_sampling_support_replay": False,
            "rollout_top_p": 1.0,
            "rollout_top_k": -1,
            "rollout_temperature": 1.0,
            "rollout_max_response_len": 16,
            "max_seq_len": None,
            "use_rollout_routing_replay": False,
            "use_rollout_indexer_replay": False,
            "use_output_store": False,
            "rollout_top_logprobs_num": 0,
            "rollout_sampling_logprobs_mode": "selected",
            "lora_rank": 0,
            "lora_adapter_path": None,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    ("rollout_top_p", "rollout_top_k", "request_top_p", "request_top_k", "expected"),
    [
        (0.95, 32, 0.95, 32, True),
        (1.0, 32, 1.0, 32, True),
        (1.0, -1, 1.0, -1, False),
    ],
)
def test_generate_payload_requests_the_sampling_mask_under_replay(
    rollout_top_p,
    rollout_top_k,
    request_top_p,
    request_top_k,
    expected,
):
    args = SimpleNamespace(
        use_rollout_sampling_support_replay=expected,
        rollout_top_p=rollout_top_p,
        rollout_top_k=rollout_top_k,
        rollout_temperature=1.0,
        rollout_max_response_len=16,
        max_seq_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_output_store=False,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        lora_rank=0,
        lora_adapter_path=None,
    )

    payload, halt_status = compute_request_payload(
        args,
        input_ids=[1, 2],
        sampling_params={
            "max_new_tokens": 4,
            "top_p": request_top_p,
            "top_k": request_top_k,
            "temperature": 1.0,
        },
    )

    assert halt_status is None
    assert payload.get("return_sampling_mask", False) is expected
    assert payload.get("return_flat_raw_sampling_mask", False) is expected
    assert payload.get("return_flat_raw_sampling_mask_b64", False) is expected


@pytest.mark.parametrize(
    ("fields", "flat"),
    [
        ({"return_sampling_mask": True}, True),
        ({"return_sampling_mask": True, OUTPUT_STORE_REQUEST_KEY: True}, False),
        ({"return_sampling_mask": False, "return_flat_raw_sampling_mask": True}, False),
    ],
    ids=["inline", "output-store", "not-requested"],
)
def test_sampling_mask_reply_is_flat_only_when_the_mask_travels_in_it(fields, flat):
    request = dict(fields)

    configure_sampling_mask_request(request)

    assert request.get("return_flat_raw_sampling_mask", False) is flat
    assert request.get("return_flat_raw_sampling_mask_b64", False) is flat


def test_enabled_replay_accepts_request_top_k_above_default():
    args = _payload_args(use_rollout_sampling_support_replay=True)

    assert should_return_sampling_mask(args, {"top_p": 1.0, "top_k": 64, "temperature": 1.0}) is True


@pytest.mark.parametrize(
    ("sampling", "mode"),
    [
        ({"top_p": 1.0, "top_k": 32}, "selected"),
        ({"top_p": 0.9, "top_k": -1}, "selected"),
        ({"top_p": 1.0, "top_k": -1, "min_p": 0.1}, "selected"),
        ({"top_p": 1.0, "top_k": -1}, None),
    ],
    ids=["top-k", "top-p", "min-p", "unfiltered"],
)
def test_filtered_request_without_replay_asks_for_the_sampler_logprobs(sampling, mode):
    """Without replay, a request that filters its sampling returns no mask but asks for the
    sampler's own log-probs; an unfiltered request needs neither."""
    payload, halt_status = compute_request_payload(
        _payload_args(rollout_top_k=32),
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 4, "temperature": 1.0, **sampling},
    )

    assert halt_status is None
    assert "return_sampling_mask" not in payload
    assert payload.get("sampling_logprobs_mode") == mode


def test_filtered_request_without_replay_reads_the_sampler_logprobs():
    """A response to selected mode without the mask stores output_token_sampling_logprobs, one per output token."""
    sample = Sample(tokens=[1, 2], response="", response_length=0)
    output = {
        "text": "ab",
        "meta_info": {
            "output_token_logprobs": [[-1.5, 10], [-2.5, 11]],
            "output_token_sampling_logprobs": [-0.5, -1.25],
            "finish_reason": {"type": "stop"},
        },
    }

    asyncio.run(
        update_sample_from_response(
            _payload_args(sglang=make_sglang_config(speculative_algorithm=None)),
            sample,
            {"input_ids": [1, 2], "sampling_logprobs_mode": "selected"},
            output,
        )
    )

    assert sample.rollout_log_probs == [-0.5, -1.25]
    assert sample.rollout_sampling_mask is None
    with pytest.raises(ValueError, match="1 values for 2 output tokens"):
        read_selected_sampling_logprobs({"output_token_sampling_logprobs": [-0.5]}, [10, 11])


def test_support_candidates_without_replay_request_the_mask():
    """Recorded support-mode candidates need the mask from SGLang even without replay."""
    args = _payload_args(rollout_top_k=32, rollout_top_logprobs_num=64, rollout_sampling_logprobs_mode="support")

    assert should_return_sampling_mask(args, {"top_p": 0.9, "top_k": 32, "temperature": 1.0}) is True
    with pytest.raises(ValueError, match="positive request top_k"):
        should_return_sampling_mask(args, {"top_p": 0.9, "top_k": -1, "temperature": 1.0})


@pytest.mark.parametrize(
    "sampling",
    [{"top_p": 1.0, "top_k": 32}, {"top_p": 0.9, "top_k": -1}, {"top_p": 1.0, "top_k": -1, "min_p": 0.1}],
    ids=["top-k", "top-p", "min-p"],
)
def test_unbounded_run_rejects_bounded_training_request(sampling):
    args = _payload_args()

    with pytest.raises(ValueError, match="bounded training-request sampling requires bounded rollout sampling"):
        should_return_sampling_mask(args, {"temperature": 1.0, **sampling})


def test_unbounded_run_treats_an_unset_request_top_k_as_unbounded():
    args = _payload_args()

    assert should_return_sampling_mask(args, {"top_p": 1.0, "top_k": None}) is False


def test_evaluation_does_not_request_or_validate_training_sampling_support():
    args = SimpleNamespace(
        use_rollout_sampling_support_replay=True,
        rollout_top_p=0.95,
        rollout_top_k=32,
        rollout_temperature=1.0,
        rollout_max_response_len=16,
        max_seq_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_output_store=False,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        lora_rank=0,
        lora_adapter_path=None,
    )

    payload, halt_status = compute_request_payload(
        args,
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 4, "top_p": 1.0, "top_k": -1, "temperature": 0.5},
        evaluation=True,
    )

    assert halt_status is None
    assert "return_sampling_mask" not in payload
    assert "return_flat_raw_sampling_mask" not in payload


def test_training_request_temperature_must_match_actor_scoring_temperature():
    args = _payload_args(use_rollout_sampling_support_replay=True)

    with pytest.raises(ValueError, match="request temperature 0.5"):
        should_return_sampling_mask(args, {"top_p": 0.95, "top_k": 32, "temperature": 0.5})


@pytest.mark.parametrize("missing_param", ["top_p", "top_k", "temperature"])
def test_sampling_support_replay_requires_explicit_request_parameters(missing_param):
    args = _payload_args(use_rollout_sampling_support_replay=True)
    params = {"top_p": 0.95, "top_k": 32, "temperature": 1.0}
    del params[missing_param]

    with pytest.raises(ValueError, match=rf"explicit request parameters:.*{missing_param}"):
        should_return_sampling_mask(args, params)


@pytest.mark.parametrize("request_top_k", [-1, 0])
def test_training_request_top_k_must_be_positive(request_top_k):
    args = _payload_args(use_rollout_sampling_support_replay=True)

    with pytest.raises(ValueError, match="request top_k must be positive"):
        should_return_sampling_mask(
            args,
            {"top_p": 0.95, "top_k": request_top_k, "temperature": 1.0},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("frequency_penalty", 0.1),
        ("presence_penalty", 0.1),
        ("repetition_penalty", 1.1),
        ("logit_bias", {"1": 0.1}),
        ("custom_logit_processor", "serialized-processor"),
    ],
)
def test_training_request_rejects_unreplayed_logit_transform(name, value):
    args = _payload_args(use_rollout_sampling_support_replay=True)

    with pytest.raises(ValueError, match=rf"{name} is not supported"):
        should_return_sampling_mask(
            args,
            {"top_p": 0.95, "top_k": 32, "temperature": 1.0, name: value},
        )


def test_append_sampling_metadata_preserves_ragged_support_and_native_logprobs():
    sample = Sample(tokens=[1, 2])
    meta_info = _flat_sampling_mask([[10, 4, 7], [11, 3]], [-0.25, -0.5])

    log_probs = append_sampling_metadata(sample, [10, 11], meta_info)
    sample.tokens.extend([10, 11])
    sample.response_length = 2

    assert log_probs == [-0.25, -0.5]
    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4, 7, 11, 3]
    assert offsets.tolist() == [0, 3, 5]
    sample.validate()


def test_append_sampling_metadata_selects_sampled_probability_from_support_mode():
    sample = Sample(tokens=[1])
    meta_info = _flat_sampling_mask([[10, 4, 7], [11, 3]], [[-1.25, -0.625, -2.0], [-0.375, -1.125]])

    log_probs = append_sampling_metadata(sample, [4, 11], meta_info, sampling_logprobs_mode="support")

    assert log_probs == [-0.625, -0.375]
    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4, 7, 11, 3]
    assert offsets.tolist() == [0, 3, 5]


def test_append_sampling_metadata_rejects_misaligned_support_logprobs():
    sample = Sample(tokens=[1])
    meta_info = _flat_sampling_mask([[10, 4]], [[-0.5]])

    with pytest.raises(ValueError, match="align"):
        append_sampling_metadata(sample, [4], meta_info, sampling_logprobs_mode="support")


def test_forced_tokens_append_singleton_support_and_strip_cleanly():
    sample = Sample(
        tokens=[1, 10],
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[10, 4], offsets=[0, 2]),
    )

    append_forced_sampling_tokens(sample, [20, 21])
    sample.tokens.extend([20, 21])
    sample.response_length += 2
    sample.validate()

    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4, 20, 21]
    assert offsets.tolist() == [0, 2, 3, 4]

    tokenizer = type("Tokenizer", (), {"decode": staticmethod(lambda _: "")})()
    sample.strip_last_output_tokens(2, tokenizer)
    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4]
    assert offsets.tolist() == [0, 2]


def test_merge_sampling_masks_inserts_singleton_observation_supports():
    first = Sample(
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[10, 4], offsets=[0, 2]),
    )
    second = Sample(
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[30, 7, 8], offsets=[0, 3]),
    )

    sampling_mask = merge_sampling_masks(first, [20, 21], second)
    ids, offsets = sampling_mask._as_tensors()

    assert ids.tolist() == [10, 4, 20, 21, 30, 7, 8]
    assert offsets.tolist() == [0, 2, 3, 4, 7]


def test_append_sampling_metadata_rejects_support_without_sampled_token():
    with pytest.raises(ValueError, match="sampled token 10 is absent"):
        append_sampling_metadata(Sample(), [10], _flat_sampling_mask([[4, 7]], [-0.25]))


def test_append_sampling_metadata_rejects_nested_rows():
    nested = {"output_token_sampling_mask": [[10, 4]], "output_token_sampling_logprobs": [-0.25]}

    with pytest.raises(ValueError, match="missing output_token_sampling_mask_flat_b64"):
        append_sampling_metadata(Sample(), [10], nested)


def test_abort_before_sampling_does_not_require_sampling_metadata():
    sample = Sample()

    assert append_sampling_metadata(sample, [], {"finish_reason": {"type": "abort"}}) == []
    assert len(sample.rollout_sampling_mask) == 0


def test_no_output_flat_reply_appends_an_empty_mask():
    sample = Sample()

    assert append_sampling_metadata(sample, [], _flat_sampling_mask([], [])) == []
    assert len(sample.rollout_sampling_mask) == 0


@pytest.mark.parametrize(
    ("mode", "key"),
    [("selected", "output_token_sampling_logprobs"), ("support", "output_token_sampling_selected_logprobs")],
)
def test_read_sampled_logprobs_reads_one_value_per_sampled_token(mode, key):
    # The support rows went to the output store; the other key must not be read.
    other = "output_token_sampling_logprobs" if mode == "support" else "output_token_sampling_selected_logprobs"
    meta_info = {key: [-0.25, -1.5], other: [[0.0]]}

    assert read_sampled_logprobs(meta_info, [4, 7], sampling_logprobs_mode=mode) == [-0.25, -1.5]
    with pytest.raises(StorePlanError, match=f"{key} has 2 values for 3 output tokens"):
        read_sampled_logprobs(meta_info, [4, 7, 9], sampling_logprobs_mode=mode)
    with pytest.raises(StorePlanError, match=f"{key} has 0 values for 1 output tokens"):
        read_sampled_logprobs({}, [4], sampling_logprobs_mode=mode)


def test_read_sampled_logprobs_of_no_output_are_empty():
    # A no-output abort returns no log-probs at all.
    assert read_sampled_logprobs({}, [], sampling_logprobs_mode="support") == []
    with pytest.raises(ValueError, match="Unsupported sampling logprobs mode"):
        read_sampled_logprobs({}, [], sampling_logprobs_mode="other")
