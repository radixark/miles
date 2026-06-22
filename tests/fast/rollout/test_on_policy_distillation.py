import asyncio
import math
from argparse import Namespace

import aiohttp
import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

import miles.rollout.on_policy_distillation as opd
from miles.rollout.on_policy_distillation import (
    _compute_topk_reverse_kl,
    _is_retryable_scoring_error,
    _mixture_log_probs,
    _mixture_logprob_maps,
    _per_position_ids,
    _score_payload,
    _scoring_timeout,
    _tail_bucket_reverse_kl,
    _teacher_targets_for_sample,
    parse_teacher_urls,
    post_process_rewards,
    reward_func,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=60, suite="stage-a-cpu")


def _entry(prob: float, token_id: int):
    return [math.log(prob), token_id]


def _args(strategy: str, weight_mode: str = "student_p"):
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
    )


def _sample():
    return Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 4), _entry(0.3, 5)],
            ]
        },
    )


def _teacher_payload():
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.5, 2), _entry(0.5, 3)],
                    [_entry(0.8, 4), _entry(0.2, 6)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.7, 2)],
                    [_entry(0.4, 4), _entry(0.6, 5)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.4, 2), _entry(0.2, 3)],
                    [_entry(0.7, 4), _entry(0.1, 6)],
                ]
            }
        },
    }


def test_topk_only_student_uses_student_probability_weights():
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_intersection_uses_overlap_only():
    reverse_kl = _compute_topk_reverse_kl(_args("intersection", "none"), _sample(), _teacher_payload())

    assert reverse_kl.tolist() == pytest.approx(
        [
            math.log(0.4 / 0.5),
            math.log(0.7 / 0.8),
        ]
    )


def test_topk_only_teacher_does_not_need_student_top_logprobs():
    sample = Sample(tokens=[10, 11, 12], response_length=2)

    reverse_kl = _compute_topk_reverse_kl(_args("only-teacher"), sample, _teacher_payload())

    expected_0 = (2 / 3) * math.log(0.4 / 0.5) + (1 / 3) * math.log(0.2 / 0.5)
    expected_1 = (7 / 8) * math.log(0.7 / 0.8) + (1 / 8) * math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_xor_uses_symmetric_difference_without_normalization():
    reverse_kl = _compute_topk_reverse_kl(_args("xor", "none"), _sample(), _teacher_payload())

    expected_0 = math.log(0.6 / 0.3) + math.log(0.2 / 0.5)
    expected_1 = math.log(0.3 / 0.6) + math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_per_position_ids_pads_prompt_and_keeps_response_order():
    # Two response positions, each with two top-k entries [logprob, token_id].
    student_top = [[_entry(0.6, 5), _entry(0.4, 7)], [_entry(0.7, 9), _entry(0.3, 11)]]
    per_pos = _per_position_ids(student_top, prompt_len=3)
    # 3 empty prompt slots, then response positions with their own token ids.
    assert per_pos == [[], [], [], [5, 7], [9, 11]]
    # Aligns with the existing _trim_input_field extraction values[1:][-R:]: for a
    # length-5 response, indices 3,4 are the response positions.
    values = list(range(5))
    assert values[1:][-2:] == [3, 4]
    assert per_pos[3] == [5, 7] and per_pos[4] == [9, 11]


def test_score_payload_routes_per_position_vs_flat():
    flat = _score_payload([1, 2, 3], token_ids=[5, 7])
    assert flat["token_ids_logprob"] == [5, 7]
    assert "token_ids_logprob_positions" not in flat

    per_pos = _score_payload([1, 2, 3], token_ids_positions=[[], [5, 7], [9, 11]])
    assert per_pos["token_ids_logprob_positions"] == [[], [5, 7], [9, 11]]
    assert "token_ids_logprob" not in per_pos


# ---------------------------------------------------------------------------
# Multi-teacher routing (--opd-teacher-urls)
# ---------------------------------------------------------------------------


def _routing_args(urls=None, key="opd_teacher", rm_url="http://single-teacher/generate"):
    return Namespace(opd_teacher_urls=urls, opd_teacher_key=key, rm_url=rm_url)


def _tagged_sample(metadata=None):
    return Sample(tokens=[1, 2, 3], response_length=2, metadata=metadata or {})


def test_parse_teacher_urls_parses_names_and_keeps_equals_in_url():
    url_map = parse_teacher_urls(["math=http://h1:30001/generate", "code=http://h2:30002/generate?tag=a=b"])
    assert url_map == {
        "math": [("http://h1:30001/generate", 1.0)],
        "code": [("http://h2:30002/generate?tag=a=b", 1.0)],
    }


def test_parse_teacher_urls_empty_or_none_gives_empty_map():
    assert parse_teacher_urls(None) == {}
    assert parse_teacher_urls([]) == {}


@pytest.mark.parametrize("bad", ["math", "=http://h1/generate", "math=", "  =  "])
def test_parse_teacher_urls_rejects_malformed_entries(bad):
    with pytest.raises(ValueError, match="expected NAME=URL"):
        parse_teacher_urls([bad])


def test_parse_teacher_urls_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate teacher name"):
        parse_teacher_urls(["math=http://h1/generate", "math=http://h2/generate"])


def test_parse_teacher_urls_parses_ensemble_groups_and_weights():
    url_map = parse_teacher_urls(["ens=http://h1/generate,http://h2/generate@2.5", "solo=http://h3/generate"])
    assert url_map == {
        "ens": [("http://h1/generate", 1.0), ("http://h2/generate", 2.5)],
        "solo": [("http://h3/generate", 1.0)],
    }


def test_parse_teacher_urls_keeps_at_sign_in_userinfo_urls():
    # rpartition on the last '@': a suffix that is not a float belongs to the URL.
    url_map = parse_teacher_urls(["m=http://user:pw@h1/generate"])
    assert url_map == {"m": [("http://user:pw@h1/generate", 1.0)]}
    url_map = parse_teacher_urls(["m=http://user:pw@h1/generate@0.5"])
    assert url_map == {"m": [("http://user:pw@h1/generate", 0.5)]}


@pytest.mark.parametrize("bad_weight", ["0", "-1", "inf", "nan"])
def test_parse_teacher_urls_rejects_non_positive_or_non_finite_weights(bad_weight):
    with pytest.raises(ValueError, match="positive finite"):
        parse_teacher_urls([f"m=http://h1/generate@{bad_weight}"])


def test_parse_teacher_urls_rejects_empty_group_member():
    with pytest.raises(ValueError, match="Empty teacher URL"):
        parse_teacher_urls(["ens=http://h1/generate,,http://h2/generate"])


def test_parse_teacher_urls_rejects_duplicate_url_within_group():
    with pytest.raises(ValueError, match="Duplicate URL within teacher group"):
        parse_teacher_urls(["ens=http://h1/generate,http://h1/generate@2.0"])


def test_parse_teacher_urls_rejects_non_http_members_at_startup():
    # A comma inside a member URL splits it into fragments; the scheme check turns
    # that into a startup error instead of a mid-rollout invalid-endpoint failure.
    with pytest.raises(ValueError, match="must start with"):
        parse_teacher_urls(["math=http://h1/generate?tags=a,b"])
    with pytest.raises(ValueError, match="must start with"):
        parse_teacher_urls(["math=h1:30001/generate"])


def test_routing_unset_map_falls_back_to_rm_url():
    args = _routing_args(urls=None)
    sample = _tagged_sample({"opd_teacher": "math"})
    assert _teacher_targets_for_sample(args, sample) == [("http://single-teacher/generate", 1.0)]


def test_routing_by_metadata_name():
    args = _routing_args(urls=["math=http://h1/generate", "code=http://h2/generate"])
    assert _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "math"})) == [("http://h1/generate", 1.0)]
    assert _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "code"})) == [("http://h2/generate", 1.0)]


def test_routing_resolves_ensemble_group():
    args = _routing_args(urls=["math=http://h1/generate@1.0,http://h2/generate@3.0"])
    assert _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "math"})) == [
        ("http://h1/generate", 1.0),
        ("http://h2/generate", 3.0),
    ]


def test_routing_respects_custom_metadata_key():
    args = _routing_args(urls=["math=http://h1/generate"], key="task")
    assert _teacher_targets_for_sample(args, _tagged_sample({"task": "math"})) == [("http://h1/generate", 1.0)]


def test_routing_missing_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_targets_for_sample(args, _tagged_sample({})) == [("http://h3/generate", 1.0)]


def test_routing_unknown_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "physics"})) == [
        ("http://h3/generate", 1.0)
    ]


def test_routing_unknown_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="matches no --opd-teacher-urls name"):
        _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "physics"}))


def test_routing_missing_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="missing teacher key"):
        _teacher_targets_for_sample(args, _tagged_sample({}))


# ---------------------------------------------------------------------------
# Teacher ensembles: probability-space mixtures
# ---------------------------------------------------------------------------


def test_mixture_log_probs_is_arithmetic_mean_in_probability_space():
    teacher_a = torch.tensor([0.2, 0.8]).log()
    teacher_b = torch.tensor([0.6, 0.4]).log()
    mix = _mixture_log_probs([teacher_a, teacher_b], [1.0, 1.0])
    assert mix.tolist() == pytest.approx([math.log(0.4), math.log(0.6)], rel=1e-6)


def test_mixture_log_probs_respects_weights():
    teacher_a = torch.tensor([0.2, 0.8]).log()
    teacher_b = torch.tensor([0.6, 0.4]).log()
    mix = _mixture_log_probs([teacher_a, teacher_b], [1.0, 3.0])
    # (0.2 + 3*0.6)/4 = 0.5 and (0.8 + 3*0.4)/4 = 0.5
    assert mix.tolist() == pytest.approx([math.log(0.5), math.log(0.5)], rel=1e-6)


def test_mixture_logprob_maps_mixes_per_token_id():
    maps_a = [{1: math.log(0.2), 2: math.log(0.6)}]
    maps_b = [{1: math.log(0.4), 2: math.log(0.8)}]
    mixed = _mixture_logprob_maps([maps_a, maps_b], [1.0, 1.0])
    assert mixed[0][1] == pytest.approx(math.log(0.3), rel=1e-9)
    assert mixed[0][2] == pytest.approx(math.log(0.7), rel=1e-9)


def test_mixture_logprob_maps_raises_on_missing_token_id():
    maps_a = [{1: math.log(0.2), 2: math.log(0.6)}]
    maps_b = [{1: math.log(0.4)}]
    with pytest.raises(ValueError, match="missing logprob for token id"):
        _mixture_logprob_maps([maps_a, maps_b], [1.0, 1.0])


def _ensemble_teacher_response(probs_by_position):
    # probs_by_position: list (per response position) of {token_id: prob}.
    return {
        "meta_info": {
            "input_token_ids_logprobs": [None]
            + [[_entry(p, tid) for tid, p in position.items()] for position in probs_by_position]
        }
    }


def test_topk_ensemble_uses_mixture_of_teachers():
    # Per-teacher maps whose uniform mixture equals the single-teacher fixture
    # (_teacher_payload input_token_ids_logprobs), so the expected KL matches
    # test_topk_only_student_uses_student_probability_weights exactly.
    payload = {
        "teachers": [
            _ensemble_teacher_response([{1: 0.2, 2: 0.6}, {4: 0.5, 5: 0.7}]),
            _ensemble_teacher_response([{1: 0.4, 2: 0.8}, {4: 0.3, 5: 0.5}]),
        ],
        "teacher_weights": [1.0, 1.0],
    }
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), payload)

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)
    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1], rel=1e-5)


def test_topk_ensemble_rejects_non_student_strategies():
    payload = {
        "teachers": [
            _ensemble_teacher_response([{1: 0.2, 2: 0.6}, {4: 0.5, 5: 0.7}]),
            _ensemble_teacher_response([{1: 0.4, 2: 0.8}, {4: 0.3, 5: 0.5}]),
        ],
        "teacher_weights": [1.0, 1.0],
    }
    with pytest.raises(ValueError, match="require --opd-top-k-strategy only-student"):
        _compute_topk_reverse_kl(_args("union"), _sample(), payload)


# ---------------------------------------------------------------------------
# Tail-mass bucket (--opd-topk-tail-bucket)
# ---------------------------------------------------------------------------


def _tail_args(strategy: str = "only-student"):
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode="student_p",
        opd_topk_tail_bucket=True,
    )


def test_tail_bucket_reverse_kl_matches_hand_computed_bucket_kl():
    # Buckets: {id1: 0.6, id2: 0.3, tail: 0.1} vs {id1: 0.3, id2: 0.5, tail: 0.2}.
    student_logps = [math.log(0.6), math.log(0.3)]
    teacher_logps = [math.log(0.3), math.log(0.5)]
    expected = 0.6 * math.log(0.6 / 0.3) + 0.3 * math.log(0.3 / 0.5) + 0.1 * (math.log(0.1) - math.log(0.2))
    assert _tail_bucket_reverse_kl(student_logps, teacher_logps) == pytest.approx(expected, rel=1e-9)


def test_tail_bucket_skips_tail_when_student_mass_at_least_one():
    student_logps = [math.log(0.8), math.log(0.3)]
    teacher_logps = [math.log(0.4), math.log(0.4)]
    expected = 0.8 * math.log(0.8 / 0.4) + 0.3 * math.log(0.3 / 0.4)
    assert _tail_bucket_reverse_kl(student_logps, teacher_logps) == pytest.approx(expected, rel=1e-9)


def test_tail_bucket_floors_vanishing_teacher_tail():
    student_logps = [math.log(0.5)]
    teacher_logps = [math.log(1.0)]
    # The teacher tail mass rounds to 0 and is floored to TAIL_PROB_FLOOR; the exact
    # floored value depends on float64 rounding of 1 - 1e-12, hence the loose rel tol.
    expected = 0.5 * math.log(0.5 / 1.0) + 0.5 * (math.log1p(-0.5) - math.log(opd.TAIL_PROB_FLOOR))
    result = _tail_bucket_reverse_kl(student_logps, teacher_logps)
    assert math.isfinite(result)
    assert result == pytest.approx(expected, rel=1e-4)


def test_tail_bucket_empty_token_set_contributes_zero():
    assert _tail_bucket_reverse_kl([], []) == pytest.approx(0.0, abs=1e-12)


def test_topk_tail_bucket_rejects_mixed_source_strategies():
    # only-teacher/union/xor fill student logprobs from a separate
    # rescoring pass, so the k+1 bucket masses do not come from one softmax.
    with pytest.raises(ValueError, match="only-student.*intersection"):
        _compute_topk_reverse_kl(_tail_args("union"), _sample(), _teacher_payload())


def test_topk_ensemble_with_tail_bucket_and_weights_matches_hand_computed_mixture_kl():
    # The configuration the ensemble example ships: weighted ensemble (2:1) + tail
    # bucket. Mixture probs are the weighted average of member probs, and the
    # mixture tail is the weighted average of member tails over the same ids.
    sample = Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.3, 2)],
                [_entry(0.5, 4), _entry(0.2, 5)],
            ]
        },
    )
    payload = {
        "teachers": [
            _ensemble_teacher_response([{1: 0.2, 2: 0.6}, {4: 0.5, 5: 0.1}]),
            _ensemble_teacher_response([{1: 0.5, 2: 0.4}, {4: 0.2, 5: 0.4}]),
        ],
        "teacher_weights": [2.0, 1.0],
    }
    reverse_kl = _compute_topk_reverse_kl(_tail_args(), sample, payload)

    t0_id1 = (2.0 * 0.2 + 1.0 * 0.5) / 3.0
    t0_id2 = (2.0 * 0.6 + 1.0 * 0.4) / 3.0
    t1_id4 = (2.0 * 0.5 + 1.0 * 0.2) / 3.0
    t1_id5 = (2.0 * 0.1 + 1.0 * 0.4) / 3.0
    expected_0 = (
        0.6 * math.log(0.6 / t0_id1)
        + 0.3 * math.log(0.3 / t0_id2)
        + 0.1 * (math.log(0.1) - math.log(1.0 - t0_id1 - t0_id2))
    )
    expected_1 = (
        0.5 * math.log(0.5 / t1_id4)
        + 0.2 * math.log(0.2 / t1_id5)
        + 0.3 * (math.log(0.3) - math.log(1.0 - t1_id4 - t1_id5))
    )
    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1], rel=1e-5)


def test_topk_tail_bucket_end_to_end():
    sample = Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.3, 2)],
                [_entry(0.5, 4), _entry(0.2, 5)],
            ]
        },
    )
    payload = {
        "teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.5, 2)],
                    [_entry(0.4, 4), _entry(0.1, 5)],
                ]
            }
        }
    }
    reverse_kl = _compute_topk_reverse_kl(_tail_args(), sample, payload)

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.3 * math.log(0.3 / 0.5) + 0.1 * (math.log(0.1) - math.log(0.2))
    expected_1 = 0.5 * math.log(0.5 / 0.4) + 0.2 * math.log(0.2 / 0.1) + 0.3 * (math.log(0.3) - math.log(0.5))
    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1], rel=1e-5)


# ---------------------------------------------------------------------------
# Scoring robustness: timeout fallback and bounded retry
# ---------------------------------------------------------------------------


def test_scoring_timeout_prefers_dedicated_flag_over_router_timeout():
    assert _scoring_timeout(Namespace(opd_scoring_timeout_secs=30.0, sglang_router_request_timeout_secs=600)) == 30.0
    assert _scoring_timeout(Namespace(opd_scoring_timeout_secs=None, sglang_router_request_timeout_secs=600)) == 600
    assert _scoring_timeout(Namespace()) is None


def test_is_retryable_scoring_error_classification():
    assert _is_retryable_scoring_error(asyncio.TimeoutError())
    assert _is_retryable_scoring_error(aiohttp.ClientConnectionError())
    assert _is_retryable_scoring_error(aiohttp.ClientResponseError(request_info=None, history=(), status=503))
    assert not _is_retryable_scoring_error(aiohttp.ClientResponseError(request_info=None, history=(), status=400))
    assert not _is_retryable_scoring_error(ValueError("missing logprob"))


async def _no_sleep(_secs):
    return None


def test_post_json_retries_transient_failure_once(monkeypatch):
    calls = []

    async def flaky(url, payload, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise aiohttp.ClientConnectionError("transient")
        return {"ok": True}

    monkeypatch.setattr(opd, "_post_json_once", flaky)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    assert asyncio.run(opd._post_json("http://t/generate", {})) == {"ok": True}
    assert len(calls) == 2


def test_post_json_does_not_retry_client_errors(monkeypatch):
    calls = []

    async def always_400(url, payload, timeout):
        calls.append(url)
        raise aiohttp.ClientResponseError(request_info=None, history=(), status=400)

    monkeypatch.setattr(opd, "_post_json_once", always_400)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(opd._post_json("http://t/generate", {}))
    assert len(calls) == 1


def test_post_json_raises_after_retries_exhausted(monkeypatch):
    calls = []

    async def always_down(url, payload, timeout):
        calls.append(url)
        raise aiohttp.ClientConnectionError("down")

    monkeypatch.setattr(opd, "_post_json_once", always_down)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    with pytest.raises(aiohttp.ClientConnectionError):
        asyncio.run(opd._post_json("http://t/generate", {}))
    assert len(calls) == opd.SCORING_MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# reward_func fan-out and post_process_rewards ensemble handling
# ---------------------------------------------------------------------------


def _sampled_token_response(logprobs):
    # input_token_logprobs: leading placeholder + one [logprob, token_id] per token.
    return {"meta_info": {"input_token_logprobs": [[0.0, 0]] + [[lp, 0] for lp in logprobs]}}


def test_reward_func_fans_out_to_every_ensemble_member(monkeypatch):
    posted = []

    async def fake_post(url, payload, timeout_secs=None):
        posted.append((url, payload))
        return _sampled_token_response([math.log(0.5), math.log(0.5)])

    monkeypatch.setattr(opd, "_post_json", fake_post)
    args = Namespace(
        opd_log_prob_top_k=0,
        opd_teacher_urls=["default=http://h1/generate,http://h2/generate@2.0"],
        opd_teacher_key="opd_teacher",
        rm_url=None,
    )
    sample = Sample(tokens=[10, 11, 12], response_length=2, metadata={})
    reward = asyncio.run(reward_func(args, sample))

    assert [url for url, _ in posted] == ["http://h1/generate", "http://h2/generate"]
    assert posted[0][1] == posted[1][1]  # every member scores the identical payload
    assert reward["teacher_weights"] == [1.0, 2.0]
    assert len(reward["teachers"]) == 2


def test_reward_func_single_teacher_keeps_raw_response_shape(monkeypatch):
    async def fake_post(url, payload, timeout_secs=None):
        return _sampled_token_response([math.log(0.5), math.log(0.5)])

    monkeypatch.setattr(opd, "_post_json", fake_post)
    args = Namespace(opd_log_prob_top_k=0, opd_teacher_urls=None, rm_url="http://t/generate")
    sample = Sample(tokens=[10, 11, 12], response_length=2, metadata={})
    reward = asyncio.run(reward_func(args, sample))
    assert "teachers" not in reward and "meta_info" in reward


def test_reward_func_topk_ensemble_scores_every_member_at_student_ids(monkeypatch):
    posted = []

    async def fake_post(url, payload, timeout_secs=None):
        posted.append((url, payload))
        return {"meta_info": {"input_token_ids_logprobs": [None, [], []]}}

    monkeypatch.setattr(opd, "_post_json", fake_post)
    args = Namespace(
        opd_log_prob_top_k=2,
        opd_top_k_strategy="only-student",
        opd_teacher_urls=["default=http://h1/generate,http://h2/generate"],
        opd_teacher_key="opd_teacher",
        opd_topk_per_position=False,
        rm_url=None,
    )
    sample = _sample()
    reward = asyncio.run(reward_func(args, sample))

    assert [url for url, _ in posted] == ["http://h1/generate", "http://h2/generate"]
    # Every member is scored at the same student top-k ids (global union transport).
    assert posted[0][1]["token_ids_logprob"] == [1, 2, 4, 5]
    assert posted[0][1] == posted[1][1]
    assert reward["teacher_weights"] == [1.0, 1.0]
    assert len(reward["teachers"]) == 2


def test_post_process_rewards_mixes_ensemble_sampled_token_logprobs():
    args = Namespace(opd_log_prob_top_k=0, reward_key="")
    sample = Sample(tokens=[10, 11, 12], response_length=2, metadata={})
    sample.reward = {
        "teachers": [
            _sampled_token_response([math.log(0.2), math.log(0.8)]),
            _sampled_token_response([math.log(0.6), math.log(0.4)]),
        ],
        "teacher_weights": [1.0, 1.0],
    }
    post_process_rewards(args, [sample])
    assert sample.teacher_log_probs.tolist() == pytest.approx([math.log(0.4), math.log(0.6)], rel=1e-6)


def test_post_process_rewards_frees_student_top_logprob_metadata():
    args = Namespace(
        opd_log_prob_top_k=2,
        opd_top_k_strategy="only-student",
        opd_reward_weight_mode="student_p",
        reward_key="",
    )
    sample = _sample()
    sample.reward = _teacher_payload()
    post_process_rewards(args, [sample])
    assert sample.opd_reverse_kl is not None
    assert "opd_student_top_logprobs" not in sample.metadata
