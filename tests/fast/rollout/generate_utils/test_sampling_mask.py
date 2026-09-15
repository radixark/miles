from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.rollout.generate_utils.sampling_mask import (
    append_forced_sampling_tokens,
    append_sampling_metadata,
    merge_sampling_masks,
    should_return_sampling_mask,
)
from miles.utils.sampling_mask import RolloutSamplingMask, top_p_sampling_replay_enabled
from miles.utils.types import Sample


@pytest.mark.parametrize(
    ("rollout_top_p", "rollout_top_k", "request_top_p", "request_top_k", "expected"),
    [
        (0.95, 32, 0.95, 32, True),
        (1.0, 32, 1.0, 32, False),
        (1.0, -1, 1.0, -1, False),
    ],
)
def test_generate_payload_automatically_requests_sampling_mask(
    rollout_top_p,
    rollout_top_k,
    request_top_p,
    request_top_k,
    expected,
):
    args = SimpleNamespace(
        rollout_top_p=rollout_top_p,
        rollout_top_k=rollout_top_k,
        rollout_temperature=1.0,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
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


@pytest.mark.parametrize(("rollout_top_p", "request_top_p"), [(1.0, 0.95)])
def test_training_request_must_match_top_p_replay_mode(
    rollout_top_p,
    request_top_p,
):
    args = SimpleNamespace(rollout_top_p=rollout_top_p, rollout_top_k=32, rollout_temperature=1.0)

    with pytest.raises(ValueError, match="requires --rollout-top-p < 1"):
        should_return_sampling_mask(
            args,
            {"top_p": request_top_p, "top_k": 32, "temperature": 1.0},
        )


def test_enabled_replay_accepts_top_k_only_request():
    args = SimpleNamespace(rollout_top_p=0.95, rollout_top_k=32, rollout_temperature=1.0)

    assert should_return_sampling_mask(args, {"top_p": 1.0, "top_k": 32, "temperature": 1.0}) is True


def test_top_k_and_min_p_do_not_enable_sampling_mask_replay():
    args = SimpleNamespace(rollout_top_p=1.0, rollout_top_k=32, rollout_temperature=1.0)

    assert should_return_sampling_mask(args, {"top_p": 1.0, "top_k": 32, "min_p": 0.1}) is False


def test_evaluation_does_not_request_or_validate_training_sampling_support():
    args = SimpleNamespace(
        rollout_top_p=0.95,
        rollout_top_k=32,
        rollout_temperature=1.0,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
    )

    payload, halt_status = compute_request_payload(
        args,
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 4, "top_p": 1.0, "top_k": -1, "temperature": 0.5},
        evaluation=True,
    )

    assert halt_status is None
    assert "return_sampling_mask" not in payload


def test_training_request_temperature_must_match_actor_scoring_temperature():
    args = SimpleNamespace(rollout_top_p=0.95, rollout_top_k=32, rollout_temperature=1.0)

    with pytest.raises(ValueError, match="request temperature 0.5"):
        should_return_sampling_mask(args, {"top_p": 0.95, "top_k": 32, "temperature": 0.5})


@pytest.mark.parametrize("missing_param", ["top_p", "top_k", "temperature"])
def test_top_p_sampling_replay_requires_explicit_request_parameters(missing_param):
    args = SimpleNamespace(rollout_top_p=0.95, rollout_top_k=32, rollout_temperature=1.0)
    params = {"top_p": 0.95, "top_k": 32, "temperature": 1.0}
    del params[missing_param]

    with pytest.raises(ValueError, match=rf"explicit request parameters:.*{missing_param}"):
        should_return_sampling_mask(args, params)


@pytest.mark.parametrize("request_top_k", [-1, 33])
def test_training_request_top_k_must_fit_configured_bound(request_top_k):
    args = SimpleNamespace(rollout_top_p=0.95, rollout_top_k=32, rollout_temperature=1.0)

    with pytest.raises(ValueError, match=r"request top_k must be in \[1, 32\]"):
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
        ("custom_logit_processor", "processor"),
    ],
)
def test_training_request_rejects_unreplayed_logit_transform(name, value):
    args = SimpleNamespace(rollout_top_p=0.95, rollout_top_k=32, rollout_temperature=1.0)

    with pytest.raises(ValueError, match=rf"{name} is not supported"):
        should_return_sampling_mask(
            args,
            {"top_p": 0.95, "top_k": 32, "temperature": 1.0, name: value},
        )


@pytest.mark.parametrize(
    ("top_p", "top_k", "expected"),
    [(0.95, -1, True), (1.0, 32, False), (1.0, -1, False)],
)
def test_top_p_sampling_replay_enabled_only_by_top_p(top_p, top_k, expected):
    args = SimpleNamespace(rollout_top_p=top_p, rollout_top_k=top_k)
    assert top_p_sampling_replay_enabled(args) is expected


def test_append_sampling_metadata_preserves_ragged_support_and_native_logprobs():
    sample = Sample(tokens=[1, 2])
    meta_info = {
        "output_token_sampling_mask": [[10, 4, 7], [11, 3]],
        "output_token_sampling_logprobs": [-0.25, -0.5],
    }

    log_probs = append_sampling_metadata(sample, [10, 11], meta_info)
    sample.tokens.extend([10, 11])
    sample.response_length = 2

    assert log_probs == [-0.25, -0.5]
    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4, 7, 11, 3]
    assert offsets.tolist() == [0, 3, 5]
    sample.validate()


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
        append_sampling_metadata(
            Sample(),
            [10],
            {
                "output_token_sampling_mask": [[4, 7]],
                "output_token_sampling_logprobs": [-0.25],
            },
        )


def test_abort_before_sampling_does_not_require_sampling_metadata():
    sample = Sample()

    assert append_sampling_metadata(sample, [], {"finish_reason": {"type": "abort"}}) == []
    assert len(sample.rollout_sampling_mask) == 0
