import math
from argparse import Namespace

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.on_policy_distillation import (
    _compute_topk_reverse_kl,
    _per_position_ids,
    _score_payload,
    _teacher_url_for_sample,
    parse_teacher_urls,
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
        "math": "http://h1:30001/generate",
        "code": "http://h2:30002/generate?tag=a=b",
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


def test_routing_unset_map_falls_back_to_rm_url():
    args = _routing_args(urls=None)
    sample = _tagged_sample({"opd_teacher": "math"})
    assert _teacher_url_for_sample(args, sample) == "http://single-teacher/generate"


def test_routing_by_metadata_name():
    args = _routing_args(urls=["math=http://h1/generate", "code=http://h2/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "math"})) == "http://h1/generate"
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "code"})) == "http://h2/generate"


def test_routing_respects_custom_metadata_key():
    args = _routing_args(urls=["math=http://h1/generate"], key="task")
    assert _teacher_url_for_sample(args, _tagged_sample({"task": "math"})) == "http://h1/generate"


def test_routing_missing_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({})) == "http://h3/generate"


def test_routing_unknown_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "physics"})) == "http://h3/generate"


def test_routing_unknown_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="matches no --opd-teacher-urls name"):
        _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "physics"}))


def test_routing_missing_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="missing teacher key"):
        _teacher_url_for_sample(args, _tagged_sample({}))
