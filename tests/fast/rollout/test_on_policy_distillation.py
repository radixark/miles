import asyncio
import math
from argparse import Namespace

import pytest
from tests.ci.ci_register import register_cpu_ci

import miles.rollout.on_policy_distillation as opd
from miles.rollout.on_policy_distillation import (
    _compute_topk_reverse_kl,
    _score_payload,
    _scoring_post,
    _teacher_sampled_log_probs,
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


# ---------------------------------------------------------------------------
# Scoring payload: response window
# ---------------------------------------------------------------------------


def test_score_payload_materializes_only_the_response_window():
    payload = _score_payload([10, 11, 12, 13], response_length=2)

    assert payload["input_ids"] == [10, 11, 12, 13]
    # Two prompt tokens; logprobs start one token before the response window.
    assert payload["logprob_start_len"] == 1
    assert payload["sampling_params"]["max_new_tokens"] == 0


# ---------------------------------------------------------------------------
# Sampled log-prob extraction: alignment guard
# ---------------------------------------------------------------------------


def _scored_sample() -> Sample:
    return Sample(tokens=[10, 11, 12, 13], response_length=2)


def _reply(entries: list[list]) -> dict:
    return {"meta_info": {"input_token_logprobs": entries}}


def test_sampled_log_probs_match_between_full_and_window_replies():
    sample = _scored_sample()
    full_reply = _reply([[None, 10, None], [-0.5, 11, None], [-1.0, 12, None], [-2.0, 13, None]])
    window_reply = _reply([[None, 11, None], [-1.0, 12, None], [-2.0, 13, None]])

    full = _teacher_sampled_log_probs(full_reply, sample)
    window = _teacher_sampled_log_probs(window_reply, sample)

    assert full.tolist() == window.tolist() == [-1.0, -2.0]


def test_sampled_log_probs_reject_misaligned_tokens():
    reply = _reply([[None, 11, None], [-1.0, 99, None], [-2.0, 13, None]])

    with pytest.raises(ValueError, match="token alignment mismatch"):
        _teacher_sampled_log_probs(reply, _scored_sample())


# ---------------------------------------------------------------------------
# Bounded scoring transport
# ---------------------------------------------------------------------------


def _scoring_args(**overrides) -> Namespace:
    defaults = {
        "opd_scoring_timeout": 5.0,
        "opd_scoring_max_inflight": 0,
        "opd_scoring_retries": 0,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_scoring_post_retries_after_timeout_then_succeeds(monkeypatch):
    monkeypatch.setattr(opd, "_SCORING_RETRY_BACKOFF_S", 0.01)
    calls = {"count": 0}

    async def flaky_post(url, payload, max_retries=1):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("first attempt times out")
        return {"ok": True}

    monkeypatch.setattr(opd, "post", flaky_post)
    sample = _scored_sample()

    result = asyncio.run(
        _scoring_post(
            _scoring_args(opd_scoring_retries=1), "http://teacher", {"input_ids": [1]}, sample=sample, target="teacher"
        )
    )

    assert result == {"ok": True}
    assert calls["count"] == 2


def test_scoring_post_shares_one_deadline_across_retries(monkeypatch):
    monkeypatch.setattr(opd, "_SCORING_RETRY_BACKOFF_S", 0.01)
    calls = {"count": 0}

    async def hanging_post(url, payload, max_retries=1):
        calls["count"] += 1
        await asyncio.sleep(60)

    monkeypatch.setattr(opd, "post", hanging_post)

    async def run() -> float:
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(RuntimeError, match="failed after"):
            await _scoring_post(
                _scoring_args(opd_scoring_timeout=0.2, opd_scoring_retries=5),
                "http://teacher",
                {"input_ids": [1]},
                sample=_scored_sample(),
                target="teacher",
            )
        return loop.time() - start

    # Five retries share the 0.2s deadline instead of each getting a fresh one.
    assert asyncio.run(run()) < 5.0
    assert calls["count"] == 1


def test_scoring_post_bounds_inflight_requests(monkeypatch):
    state = {"current": 0, "max": 0}

    async def tracked_post(url, payload, max_retries=1):
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.01)
        state["current"] -= 1
        return {"ok": True}

    monkeypatch.setattr(opd, "post", tracked_post)
    args = _scoring_args(opd_scoring_max_inflight=1)

    async def run():
        await asyncio.gather(
            *(
                _scoring_post(args, "http://teacher", {"input_ids": [1]}, sample=_scored_sample(), target="teacher")
                for _ in range(4)
            )
        )

    asyncio.run(run())

    assert state["max"] == 1


def test_scoring_post_deadline_includes_inflight_wait(monkeypatch):
    entered_post = asyncio.Event()
    release_post = asyncio.Event()

    async def blocked_post(url, payload, max_retries=1):
        entered_post.set()
        await release_post.wait()
        return {"ok": True}

    monkeypatch.setattr(opd, "post", blocked_post)
    long_args = _scoring_args(opd_scoring_timeout=5.0, opd_scoring_max_inflight=1)
    short_args = _scoring_args(opd_scoring_timeout=0.05, opd_scoring_max_inflight=1)

    async def run():
        first = asyncio.create_task(
            _scoring_post(long_args, "http://teacher", {"input_ids": [1]}, sample=_scored_sample(), target="teacher")
        )
        await entered_post.wait()
        with pytest.raises(RuntimeError, match="failed after 0 attempt"):
            await _scoring_post(
                short_args,
                "http://teacher",
                {"input_ids": [1]},
                sample=_scored_sample(),
                target="teacher",
            )
        release_post.set()
        assert await first == {"ok": True}

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Position-blocked top-k scoring
# ---------------------------------------------------------------------------


def _blocked_top_k_args(strategy: str, *, block_size: int = 2, weight_mode: str = "student_p") -> Namespace:
    return _scoring_args(
        opd_log_prob_top_k=2,
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
        opd_top_k_scoring_block_size=block_size,
        rm_url="http://teacher/generate",
        sglang_router_ip="student",
        sglang_router_port=30000,
    )


def _blocked_sample() -> Sample:
    return Sample(
        tokens=[10, 11, 12, 13, 14, 15],
        response_length=4,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 2), _entry(0.3, 3)],
                [_entry(0.8, 4), _entry(0.2, 5)],
                [_entry(0.9, 5), _entry(0.1, 6)],
            ]
        },
    )


def _candidate_score(target: str, position: int, token_id: int) -> float:
    target_offset = 0.2 if target == "teacher" else 0.4
    return -(target_offset + 0.05 * position + 0.01 * token_id)


def _blocked_reply(
    sample: Sample,
    payload: dict,
    target: str,
    *,
    weight_version: str | None = None,
) -> dict:
    prompt_length = len(sample.tokens) - sample.response_length
    start = payload["logprob_start_len"] + 1 - prompt_length
    end = len(payload["input_ids"]) - prompt_length
    response_tokens = sample.tokens[prompt_length + start : prompt_length + end]
    candidate_ids = payload["token_ids_logprob"]
    meta_info = {
        "input_token_logprobs": [None, *[[-1.0, token_id] for token_id in response_tokens]],
        "input_token_ids_logprobs": [
            None,
            *[
                [[_candidate_score(target, position, token_id), token_id] for token_id in candidate_ids]
                for position in range(start, end)
            ],
        ],
    }
    if weight_version is not None:
        meta_info["weight_version"] = weight_version
    return {"meta_info": meta_info}


def _global_candidate_reply(sample: Sample, candidate_rows: list[list], target: str) -> dict:
    candidate_ids = sorted({int(entry[1]) for row in candidate_rows for entry in row})
    return {
        "meta_info": {
            "input_token_ids_logprobs": [
                None,
                *[
                    [[_candidate_score(target, position, token_id), token_id] for token_id in candidate_ids]
                    for position in range(sample.response_length)
                ],
            ]
        }
    }


@pytest.mark.parametrize("weight_mode", ["student_p", "teacher_p", "none"])
def test_only_student_block_scoring_matches_response_wide_union(monkeypatch, weight_mode):
    sample = _blocked_sample()
    args = _blocked_top_k_args("only-student", weight_mode=weight_mode)
    calls = []

    async def fake_scoring_post(args, url, payload, *, sample, target):
        calls.append((url, payload, target))
        return _blocked_reply(sample, payload, target)

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    blocked_payload = asyncio.run(reward_func(args, sample))
    legacy_payload = {
        "teacher": _global_candidate_reply(
            sample,
            sample.metadata["opd_student_top_logprobs"],
            "teacher",
        )
    }

    assert _compute_topk_reverse_kl(args, sample, blocked_payload).tolist() == pytest.approx(
        _compute_topk_reverse_kl(args, sample, legacy_payload).tolist()
    )
    assert [call[1]["input_ids"] for call in calls] == [
        sample.tokens[:4],
        sample.tokens[:6],
    ]
    assert [call[1]["logprob_start_len"] for call in calls] == [1, 3]
    assert [call[1]["token_ids_logprob"] for call in calls] == [[1, 2, 3], [4, 5, 6]]
    compact_rows = blocked_payload["teacher"]["meta_info"]["input_token_ids_logprobs"][1:]
    assert [len(row) for row in compact_rows] == [2, 2, 2, 2]


@pytest.mark.parametrize("weight_mode", ["student_p", "teacher_p", "none"])
def test_only_teacher_block_scoring_matches_response_wide_union(monkeypatch, weight_mode):
    sample = Sample(tokens=[10, 11, 12, 13, 14, 15], response_length=4)
    args = _blocked_top_k_args("only-teacher", weight_mode=weight_mode)
    teacher_top = [
        [_entry(0.6, 1), _entry(0.4, 2)],
        [_entry(0.7, 2), _entry(0.3, 3)],
        [_entry(0.8, 4), _entry(0.2, 5)],
        [_entry(0.9, 5), _entry(0.1, 6)],
    ]
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [None, *[[-1.0, token_id] for token_id in sample.tokens[-4:]]],
            "input_top_logprobs": [None, *teacher_top],
        }
    }
    calls = []

    async def fake_scoring_post(args, url, payload, *, sample, target):
        calls.append((url, payload, target))
        if target == "teacher":
            return teacher_response
        return _blocked_reply(sample, payload, target, weight_version="7")

    async def fake_student_weight_version(args, sample):
        return "7"

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    monkeypatch.setattr(opd, "_student_weight_version", fake_student_weight_version)

    blocked_payload = asyncio.run(reward_func(args, sample))
    legacy_payload = {
        "teacher": teacher_response,
        "student_on_teacher": _global_candidate_reply(sample, teacher_top, "student"),
    }

    assert _compute_topk_reverse_kl(args, sample, blocked_payload).tolist() == pytest.approx(
        _compute_topk_reverse_kl(args, sample, legacy_payload).tolist()
    )
    assert len(calls) == 3
    assert calls[0][2] == "teacher"
    assert "token_ids_logprob" not in calls[0][1]
    assert [call[1]["token_ids_logprob"] for call in calls[1:]] == [[1, 2, 3], [4, 5, 6]]
    assert blocked_payload["student_on_teacher"]["meta_info"]["weight_version"] == "7"


def test_only_teacher_retries_all_blocks_after_student_version_change(monkeypatch):
    sample = Sample(tokens=[10, 11, 12, 13, 14, 15], response_length=4)
    args = _blocked_top_k_args("only-teacher")
    args.opd_scoring_retries = 1
    teacher_top = [
        [_entry(0.6, 1), _entry(0.4, 2)],
        [_entry(0.7, 2), _entry(0.3, 3)],
        [_entry(0.8, 4), _entry(0.2, 5)],
        [_entry(0.9, 5), _entry(0.1, 6)],
    ]
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [None, *[[-1.0, token_id] for token_id in sample.tokens[-4:]]],
            "input_top_logprobs": [None, *teacher_top],
        }
    }
    expected_versions = iter(["7", "8"])
    block_versions = iter(["7", "8", "8", "8"])
    student_calls = []

    async def fake_student_weight_version(args, sample):
        return next(expected_versions)

    async def fake_scoring_post(args, url, payload, *, sample, target):
        if target == "teacher":
            return teacher_response
        student_calls.append(payload)
        return _blocked_reply(sample, payload, target, weight_version=next(block_versions))

    monkeypatch.setattr(opd, "_SCORING_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(opd, "_student_weight_version", fake_student_weight_version)
    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    reward_payload = asyncio.run(reward_func(args, sample))

    assert len(student_calls) == 4
    assert reward_payload["student_on_teacher"]["meta_info"]["weight_version"] == "8"
    assert [payload["token_ids_logprob"] for payload in student_calls] == [
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6],
    ]
