from types import SimpleNamespace

import numpy as np
import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.rollout.generate_utils.sampling_mask import should_return_sampling_mask
from miles.utils.replay_assembly import derive_sampling_mask
from miles.utils.replay_index import ReplayDataError, ReplayIndex, sampled_logprobs_from_meta_info
from miles.utils.types import Sample


def _parse_mask(meta_info: dict, num_output_tokens: int, *, support: bool = False) -> ReplayIndex:
    return ReplayIndex.from_meta_info(
        meta_info,
        via_store=False,
        routed_experts_rows=None,
        num_layers=None,
        indexer_topk_rows=None,
        indexer_num_streams=None,
        return_sampling_mask=True,
        support_logprobs=support,
        num_output_tokens=num_output_tokens,
        aborted=False,
    )


def _derive(index: ReplayIndex, sampled: list[int], *, support: bool = False):
    pieces = index.sampling_mask.mask_pieces({})
    return derive_sampling_mask(pieces, np.asarray(sampled, dtype=np.int64))


@pytest.mark.parametrize(
    ("rollout_top_p", "rollout_top_k", "request_top_p", "request_top_k", "expected"),
    [
        (0.95, 32, 0.95, 32, True),
        (1.0, 32, 1.0, 32, True),
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
        use_sampling_support_replay=expected,
        rollout_top_p=rollout_top_p,
        rollout_top_k=rollout_top_k,
        rollout_temperature=1.0,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_output_store=False,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
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


def test_enabled_replay_accepts_request_top_k_above_default():
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)

    assert should_return_sampling_mask(args, {"top_p": 1.0, "top_k": 64, "temperature": 1.0}) is True


def test_unbounded_run_rejects_bounded_training_request():
    args = SimpleNamespace(use_sampling_support_replay=False, rollout_temperature=1.0)

    with pytest.raises(ValueError, match="bounded training-request sampling requires bounded rollout sampling"):
        should_return_sampling_mask(args, {"top_p": 1.0, "top_k": 32, "temperature": 1.0})


def test_unbounded_run_treats_an_unset_request_top_k_as_unbounded():
    args = SimpleNamespace(use_sampling_support_replay=False, rollout_temperature=1.0)

    assert should_return_sampling_mask(args, {"top_p": 1.0, "top_k": None}) is False


def test_evaluation_does_not_request_or_validate_training_sampling_support():
    args = SimpleNamespace(
        use_sampling_support_replay=True,
        rollout_top_p=0.95,
        rollout_top_k=32,
        rollout_temperature=1.0,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_output_store=False,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
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
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)

    with pytest.raises(ValueError, match="request temperature 0.5"):
        should_return_sampling_mask(args, {"top_p": 0.95, "top_k": 32, "temperature": 0.5})


@pytest.mark.parametrize("missing_param", ["top_p", "top_k", "temperature"])
def test_sampling_support_replay_requires_explicit_request_parameters(missing_param):
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)
    params = {"top_p": 0.95, "top_k": 32, "temperature": 1.0}
    del params[missing_param]

    with pytest.raises(ValueError, match=rf"explicit request parameters:.*{missing_param}"):
        should_return_sampling_mask(args, params)


@pytest.mark.parametrize("request_top_k", [-1, 0])
def test_training_request_top_k_must_be_positive(request_top_k):
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)

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
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)

    with pytest.raises(ValueError, match=rf"{name} is not supported"):
        should_return_sampling_mask(
            args,
            {"top_p": 0.95, "top_k": 32, "temperature": 1.0, name: value},
        )


def test_sampling_mask_preserves_ragged_support_and_native_logprobs():
    meta_info = {
        "output_token_sampling_mask": [[10, 4, 7], [11, 3]],
        "output_token_sampling_logprobs": [-0.25, -0.5],
    }
    index = _parse_mask(meta_info, 2)
    sample = Sample(tokens=[1, 2, 10, 11], response_length=2, replay_index=index)
    sample.validate()

    # Selected mode: the per-token log-probs are the rollout's, not part of the mask.
    assert sampled_logprobs_from_meta_info(
        meta_info, output_token_ids=[10, 11], support_logprobs=False, aborted=False
    ) == [-0.25, -0.5]
    derived = _derive(index, [10, 11])
    assert derived.support_logprobs is None
    assert derived.ids.tolist() == [10, 4, 7, 11, 3]
    assert derived.offsets.tolist() == [0, 3, 5]


def test_support_mode_keeps_support_logprobs_and_picks_the_sampled_ones():
    meta_info = {
        "output_token_sampling_mask": [[10, 4, 7], [11, 3]],
        "output_token_sampling_logprobs": [[-1.2, -0.6, -2.0], [-0.4, -1.1]],
    }
    index = _parse_mask(meta_info, 2, support=True)

    assert sampled_logprobs_from_meta_info(
        meta_info, output_token_ids=[4, 11], support_logprobs=True, aborted=False
    ) == pytest.approx([-0.6, -0.4])
    derived = _derive(index, [4, 11], support=True)
    np.testing.assert_array_equal(derived.support_logprobs, np.float32([-1.2, -0.6, -2.0, -0.4, -1.1]))
    assert derived.ids.tolist() == [10, 4, 7, 11, 3]
    assert derived.offsets.tolist() == [0, 3, 5]


def test_sampling_mask_rejects_misaligned_support_logprobs():
    meta_info = {
        "output_token_sampling_mask": [[10, 4]],
        "output_token_sampling_logprobs": [[-0.5]],
    }

    with pytest.raises(ReplayDataError, match="align"):
        _parse_mask(meta_info, 1, support=True)
    with pytest.raises(ValueError, match="align"):
        sampled_logprobs_from_meta_info(meta_info, output_token_ids=[4], support_logprobs=True, aborted=False)


@pytest.mark.parametrize("support", [[10, 7], [4, 10, 4]])
def test_support_mode_pick_requires_the_sampled_token_exactly_once(support):
    meta_info = {"output_token_sampling_mask": [support], "output_token_sampling_logprobs": [[-0.5] * len(support)]}

    with pytest.raises(ValueError, match="exactly once"):
        sampled_logprobs_from_meta_info(meta_info, output_token_ids=[4], support_logprobs=True, aborted=False)


def test_forced_tokens_append_singleton_support_and_strip_cleanly():
    index = _parse_mask({"output_token_sampling_mask": [[10, 4]], "output_token_sampling_logprobs": [-0.1]}, 1)
    sample = Sample(tokens=[1, 10], response_length=1, replay_index=index)

    sample.replay_index = sample.replay_index.append_forced_tokens(2, response_length=sample.response_length)
    sample.tokens.extend([20, 21])
    sample.response_length += 2
    sample.validate()

    derived = _derive(sample.replay_index, [10, 20, 21])
    assert derived.ids.tolist() == [10, 4, 20, 21]
    assert derived.offsets.tolist() == [0, 2, 3, 4]
    assert derived.forced.tolist() == [False, True, True]

    tokenizer = type("Tokenizer", (), {"decode": staticmethod(lambda _: "")})()
    sample.strip_last_output_tokens(2, tokenizer)
    derived = _derive(sample.replay_index, [10])
    assert derived.ids.tolist() == [10, 4]
    assert derived.offsets.tolist() == [0, 2]


def test_merge_turn_inserts_singleton_observation_supports():
    first = _parse_mask({"output_token_sampling_mask": [[10, 4]], "output_token_sampling_logprobs": [-0.1]}, 1)
    second = _parse_mask({"output_token_sampling_mask": [[30, 7, 8]], "output_token_sampling_logprobs": [-0.2]}, 1)

    merged = first.merge_turn(second, observation_tokens=2)
    derived = _derive(merged, [10, 20, 21, 30])

    assert derived.ids.tolist() == [10, 4, 20, 21, 30, 7, 8]
    assert derived.offsets.tolist() == [0, 2, 3, 4, 7]


def test_sampling_mask_rejects_support_without_sampled_token():
    index = _parse_mask({"output_token_sampling_mask": [[4, 7]], "output_token_sampling_logprobs": [-0.25]}, 1)

    with pytest.raises(ReplayDataError, match="sampled token 10 is absent"):
        _derive(index, [10])


def test_abort_before_sampling_does_not_require_sampling_metadata():
    index = ReplayIndex.from_meta_info(
        {"finish_reason": {"type": "abort"}},
        via_store=False,
        routed_experts_rows=None,
        num_layers=None,
        indexer_topk_rows=None,
        indexer_num_streams=None,
        return_sampling_mask=True,
        support_logprobs=False,
        num_output_tokens=0,
        aborted=False,
    )

    assert index.sampling_mask.num_rows == 0
