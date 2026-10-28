from argparse import Namespace

import pytest
import torch
from examples.on_policy_distillation.mopd_reasoning_gym import hooks

from miles.rollout import on_policy_distillation
from miles.utils.types import Sample

CAESAR_URL = "http://teacher-host:30000/generate"
GEOMETRY_URL = "http://teacher-host:30001/generate"


def _args() -> Namespace:
    return Namespace(
        opd_log_prob_top_k=0,
        opd_teacher_urls=[f"caesar_cipher={CAESAR_URL}", f"simple_geometry={GEOMETRY_URL}"],
        opd_teacher_key="task",
        rm_url=None,
        sglang_router_request_timeout_secs=None,
        reward_key=None,
    )


class _FakeTask:
    """reasoning_gym's score_answer for an entry whose answer is "HELLO"."""

    def score_answer(self, answer: str, entry: dict) -> float:
        return 1.0 if answer == entry["answer"] else 0.01


@pytest.fixture
def scored_by_fake_task(monkeypatch):
    monkeypatch.setattr(hooks, "_dataset", lambda task: _FakeTask())


def _sample(response: str, *, task: str = "caesar_cipher", **metadata) -> Sample:
    # two prompt tokens, two response tokens
    return Sample(
        tokens=[11, 12, 13, 14],
        response=response,
        response_length=2,
        metadata={"task": task, "entry": {"answer": "HELLO"}, **metadata},
    )


@pytest.fixture
def teacher(monkeypatch):
    """A teacher that replies the way SGLang does for logprob_start_len = prompt_len - 1."""
    calls = []

    async def fake_post_json(url, payload, timeout_secs=None):
        calls.append((url, payload))
        return {"meta_info": {"input_token_logprobs": [None, [-0.25, 13, None], [-1.5, 14, None]]}}

    monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post_json)
    return calls


class TestTaskScore:
    @pytest.mark.parametrize(
        "response, score",
        [
            ("<think>x</think><answer>HELLO</answer>", 1.0),
            ("<think>x</think><answer>WRONG</answer><answer>HELLO</answer>", 1.0),
            ("<think>x</think><answer>HELLO</answer><answer>WRONG</answer>", 0.0),
            ("<think><answer>HELLO</answer></think>no answer after thinking", 0.0),
            ("<think>still thinking <answer>HELLO</answer>", 0.0),
            ("<think>x</think><answer> HELLO \n</answer>", 1.0),
        ],
    )
    def test_only_an_exact_last_answer_after_thinking_scores(self, scored_by_fake_task, response, score):
        assert hooks.task_score(_sample(response)) == score


class TestRewardFunc:
    async def test_training_sample_is_scored_by_its_task_teacher_and_rewarded_by_the_verifier(
        self, scored_by_fake_task, teacher
    ):
        sample = _sample("<think>x</think><answer>HELLO</answer>", task="simple_geometry")

        reward = await hooks.reward_func(_args(), sample)

        assert reward == 1.0
        ((url, payload),) = teacher
        assert url == GEOMETRY_URL
        assert payload["input_ids"] == [11, 12, 13, 14]
        assert torch.equal(sample.teacher_log_probs, torch.tensor([-0.25, -1.5]))
        assert sample.metadata["opd_score_t0"] <= sample.metadata["opd_score_t1"]

    async def test_held_out_sample_gets_only_the_verifier(self, scored_by_fake_task, teacher):
        sample = _sample("<think>x</think><answer>WRONG</answer>", mopd_eval=True)

        assert await hooks.reward_func(_args(), sample) == 0.0
        assert not teacher
        assert sample.teacher_log_probs is None

    async def test_top_k_opd_is_refused(self, scored_by_fake_task, teacher):
        args = _args()
        args.opd_log_prob_top_k = 4

        with pytest.raises(AssertionError, match="sampled-token OPD"):
            await hooks.reward_func(args, _sample("<think>x</think><answer>HELLO</answer>"))


class TestPostProcessRewards:
    async def test_verifier_scores_are_logged_and_the_objective_gets_zero_task_reward(
        self, scored_by_fake_task, teacher
    ):
        args = _args()
        samples = [_sample("<think>x</think><answer>HELLO</answer>"), _sample("<think>x</think>")]
        for sample in samples:
            sample.reward = await hooks.reward_func(args, sample)

        raw_rewards, rewards = hooks.post_process_rewards(args, samples)

        assert raw_rewards == [1.0, 0.0]
        assert rewards == [0.0, 0.0]

    def test_a_sample_the_teacher_never_scored_is_refused(self):
        sample = _sample("<think>x</think>")
        sample.reward = 0.0

        with pytest.raises(ValueError, match="neither a teacher reply"):
            hooks.post_process_rewards(_args(), [sample])
