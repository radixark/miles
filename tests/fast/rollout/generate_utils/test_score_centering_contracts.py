"""Validate score-centering sampling and candidate transport contracts."""

from copy import deepcopy

import numpy as np
import pytest
from tests.fast.fixtures.score_centering_fixtures import _args, _turn

from miles.rollout.generate_utils.score_centering import (
    append_score_centering_observations,
    append_score_centering_topk,
    configure_score_centering_request,
    validate_score_centering_sample,
)
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, decode_samples_and_merge_input_sample, encode_samples
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample


def test_observation_padding_retry_and_disabled_wire() -> None:
    sample = _turn([0], [2, 3], [0.5, 0.25])
    append_score_centering_observations(sample, 2)
    sample.tokens.extend([6, 6])
    sample.response_length += 2
    sample.rollout_log_probs.extend([0.0, 0.0])
    sample.loss_mask.extend([0, 0])
    validate_score_centering_sample(sample, 3)
    sample.reset_for_retry()
    assert sample.rollout_topk_token_ids is None and sample.rollout_topk_log_probs is None
    old_fields = tuple(field for field in COMPUTED_FIELDS if not field.startswith("rollout_topk_"))
    assert encode_samples([Sample()], {}) == encode_samples([Sample()], {}, fields=old_fields)
    assert (
        decode_samples_and_merge_input_sample(encode_samples([Sample()], {}, fields=old_fields), Sample())
        .samples[0]
        .rollout_topk_token_ids
        is None
    )


@pytest.mark.parametrize("openai", [False, True])
def test_request_candidates_and_sampler_contract(openai: bool) -> None:
    request = {} if openai else {"sampling_params": {}}
    configure_score_centering_request(_args(score_centering_top_k=128), request, openai=openai)
    assert request["top_logprobs" if openai else "top_logprobs_num"] == 128
    sampling = request if openai else request["sampling_params"]
    assert sampling["temperature"] == 0.7
    sampling["top_p"] = 0.9
    with pytest.raises(ValueError, match="top_p"):
        configure_score_centering_request(_args(), request, openai=openai)
    original = deepcopy(request)
    configure_score_centering_request(_args(loss_type="policy_loss"), request, openai=openai)
    assert request == original


@pytest.mark.parametrize(
    "constraint",
    [
        {"tool_choice": "required"},
        {"tool_choice": {"type": "function", "function": {"name": "test"}}},
        {"tools": [{"type": "function", "function": {"name": "test", "strict": True}}]},
        {"response_format": {"type": "json_object"}},
    ],
)
def test_implicit_openai_grammar_constraints_are_rejected(constraint: dict) -> None:
    with pytest.raises(ValueError):
        configure_score_centering_request(_args(), constraint, openai=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rollout_top_p", 0.9),
        ("rollout_temperature", 0),
        ("rollout_top_k", 5),
        ("score_centering_top_k", 0),
        ("score_centering_tis_clip", float("inf")),
        ("score_centering_mis_low", 6),
        ("use_tis", True),
        ("advantage_estimator", "gspo"),
        ("recompute_logprobs_via_prefill", True),
        ("sglang_speculative_algorithm", "EAGLE"),
    ],
)
def test_invalid_options_fail_early(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        validate_score_centering_args(_args(**{field: value}))


@pytest.mark.parametrize("session", ["v1", "v2"])
@pytest.mark.parametrize("top_k", [21, 128])
def test_large_session_heads_require_miles_router(session: str, top_k: int) -> None:
    args = _args(use_session_server=session, score_centering_top_k=top_k, use_miles_router=False)
    with pytest.raises(ValueError, match="require --use-miles-router"):
        validate_score_centering_args(args)

    args.use_miles_router = True
    validate_score_centering_args(args)


@pytest.mark.parametrize("session", ["v1", "v2"])
def test_standard_openai_head_size_can_use_sglang_router(session: str) -> None:
    validate_score_centering_args(_args(use_session_server=session, score_centering_top_k=20, use_miles_router=False))


def test_large_native_heads_can_use_sglang_router() -> None:
    validate_score_centering_args(_args(use_session_server=None, score_centering_top_k=128, use_miles_router=False))


def test_other_losses_do_not_require_score_centering_router() -> None:
    validate_score_centering_args(
        _args(loss_type="policy_loss", use_session_server="v2", score_centering_top_k=128, use_miles_router=False)
    )


def test_missing_or_mismatched_probabilities_fail_before_training() -> None:
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_log_probs[0] -= 0.1
    with pytest.raises(ValueError, match="same sampler"):
        validate_score_centering_sample(sample, 3)
    sample.rollout_topk_token_ids[1] = -1
    with pytest.raises(ValueError, match="Every trained token"):
        validate_score_centering_sample(sample, 3)
    with pytest.raises(ValueError, match="output_top_logprobs"):
        append_score_centering_topk(Sample(response_length=1), {"output_token_logprobs": [(-1.0, 2, None)]}, 3)


@pytest.mark.parametrize("candidates", [[2, 3, -1], [-1, 3, 2], [-1, 2, -1], [-1, -1, -1]])
def test_candidate_validation_preserves_order_and_allows_repeated_padding(candidates: list[int]) -> None:
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_topk_token_ids[:] = candidates
    sample.rollout_topk_log_probs[:] = [
        -np.inf if token == -1 else np.log(0.5 if token == 2 else 0.25) for token in candidates
    ]
    if all(token == -1 for token in candidates):
        sample.loss_mask = [0, 0]
    original_ids = sample.rollout_topk_token_ids.copy()
    original_logps = sample.rollout_topk_log_probs.copy()
    validate_score_centering_sample(sample, 3)
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, original_ids)
    np.testing.assert_array_equal(sample.rollout_topk_log_probs, original_logps)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("token", [0, 3])
def test_candidate_validation_rejects_unsorted_duplicates(masked: bool, token: int) -> None:
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_topk_token_ids[1] = [token, 2, token]
    sample.rollout_topk_log_probs[1] = np.log([0.25, 0.5, 0.25])
    sample.loss_mask[1] = 0 if masked else 1
    with pytest.raises(ValueError, match="Duplicate"):
        validate_score_centering_sample(sample, 3)
