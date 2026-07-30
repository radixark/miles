import math
from argparse import Namespace

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.on_policy_self_distillation import (
    _extract_teacher_top_k,
    _score_payload,
    build_teacher_prompt,
    post_process_rewards,
    reward_func,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=30, suite="stage-a-cpu")


def _entry(probability: float, token_id: int) -> list[float | int]:
    return [math.log(probability), token_id]


def test_default_teacher_prompt_augments_the_final_user_message_without_mutating_input():
    prompt = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Solve 2 + 2."},
    ]

    teacher_prompt = build_teacher_prompt(prompt, "The answer is 4.", {})

    assert prompt[-1]["content"] == "Solve 2 + 2."
    assert teacher_prompt[0] == prompt[0]
    assert teacher_prompt[-1]["role"] == "user"
    assert "Solve 2 + 2." in teacher_prompt[-1]["content"]
    assert "The answer is 4." in teacher_prompt[-1]["content"]
    assert "independent reasoning" in teacher_prompt[-1]["content"]


def test_default_teacher_prompt_supports_raw_text():
    teacher_prompt = build_teacher_prompt("Solve 2 + 2.", "The answer is 4.", {})

    assert isinstance(teacher_prompt, str)
    assert teacher_prompt.startswith("Solve 2 + 2.")
    assert "The answer is 4." in teacher_prompt


def test_default_teacher_prompt_rejects_non_text_conversation_content():
    prompt = [{"role": "user", "content": [{"type": "image", "image": "x"}]}]

    with pytest.raises(ValueError, match="text-only"):
        build_teacher_prompt(prompt, "answer", {})


def test_default_teacher_prompt_rejects_non_text_context_messages():
    prompt = [
        {"role": "system", "content": [{"type": "image", "image": "x"}]},
        {"role": "user", "content": "Solve this."},
    ]

    with pytest.raises(ValueError, match="text-only"):
        build_teacher_prompt(prompt, "answer", {})


def test_default_teacher_prompt_rejects_tool_messages():
    prompt = [
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "Finish."},
    ]

    with pytest.raises(ValueError, match="tool messages"):
        build_teacher_prompt(prompt, "answer", {})


def test_score_payload_requests_temperature_corrected_top_k_input_scores():
    payload = _score_payload([1, 2, 3], top_k=8, temperature=1.1)

    assert payload == {
        "input_ids": [1, 2, 3],
        "sampling_params": {
            "temperature": 1.1,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 8,
    }


def test_extract_teacher_top_k_keeps_only_response_aligned_positions():
    response = {
        "meta_info": {
            "input_top_logprobs": [
                None,
                [_entry(0.7, 10), _entry(0.3, 11)],
                [_entry(0.6, 20), _entry(0.4, 21)],
                [_entry(0.8, 30), _entry(0.2, 31)],
            ]
        }
    }

    token_ids, scores = _extract_teacher_top_k(response, response_length=2, top_k=2)

    assert token_ids == [[20, 21], [30, 31]]
    assert scores[0] == pytest.approx([math.log(0.6), math.log(0.4)])
    assert scores[1] == pytest.approx([math.log(0.8), math.log(0.2)])


async def test_reward_func_scores_privileged_prompt_with_exact_student_response(monkeypatch):
    seen = {}

    async def post_json(url, payload, *, timeout_secs):
        seen.update(url=url, payload=payload, timeout_secs=timeout_secs)
        return {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.7, 10), _entry(0.3, 11)],
                    [_entry(0.6, 20), _entry(0.4, 21)],
                    [_entry(0.8, 30), _entry(0.2, 31)],
                    [_entry(0.9, 40), _entry(0.1, 41)],
                ]
            }
        }

    monkeypatch.setattr("miles.rollout.on_policy_self_distillation._post_json", post_json)
    args = Namespace(
        opsd_teacher_top_k=2,
        rollout_temperature=0.8,
        opsd_teacher_url="http://teacher/generate",
        sglang_router_request_timeout_secs=30,
    )
    sample = Sample(
        tokens=[1, 2, 30, 40],
        response_length=2,
        privileged_prompt_tokens=[7, 8, 9],
    )

    reward = await reward_func(args, sample)

    assert seen["url"] == "http://teacher/generate"
    assert seen["timeout_secs"] == 30
    assert seen["payload"]["input_ids"] == [7, 8, 9, 30, 40]
    assert seen["payload"]["sampling_params"]["temperature"] == 0.8
    assert reward["token_ids"] == [[30, 31], [40, 41]]


def test_post_process_rewards_moves_compact_teacher_support_onto_samples():
    samples = [
        Sample(
            tokens=[1, 2, 3],
            response_length=2,
            reward={
                "token_ids": [[10, 11], [20, 21]],
                "scores": [[-0.1, -2.3], [-0.2, -1.7]],
            },
        )
    ]

    raw_rewards, rewards = post_process_rewards(Namespace(opsd_teacher_top_k=2), samples)

    assert raw_rewards == [0.0]
    assert rewards == [0.0]
    assert samples[0].opsd_teacher_token_ids.tolist() == [[10, 11], [20, 21]]
    assert samples[0].opsd_teacher_scores.tolist()[0] == pytest.approx([-0.1, -2.3])
    assert samples[0].opsd_teacher_scores.tolist()[1] == pytest.approx([-0.2, -1.7])
    samples[0].validate()
