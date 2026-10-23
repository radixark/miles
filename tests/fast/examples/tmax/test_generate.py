"""TMax's reward function and pure OPD's reward post-process (generate.py)."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax"))

import generate  # noqa: E402

from miles.rollout.base_types import GenerateFnOutput  # noqa: E402

from miles.utils.types import Sample  # noqa: E402

ARGS = SimpleNamespace(reward_key=None)


def test_reward_is_the_sandbox_verdict():
    samples = [Sample(metadata={"reward": 1.0}), Sample(metadata={"reward": None}), Sample(metadata={})]
    assert asyncio.run(generate.reward_func(ARGS, samples)) == [1.0, None, None]
    assert asyncio.run(generate.reward_func(ARGS, samples[0])) == 1.0


def test_pure_opd_trains_on_zero_reward_and_logs_the_task_reward():
    samples = [Sample(reward=1.0), Sample(reward=0.0), Sample(reward=1.0)]
    raw_rewards, rewards = generate.opd_pure_rewards(ARGS, samples)
    assert raw_rewards == [1.0, 0.0, 1.0], "raw_reward (rollout/raw_reward) stays the task reward"
    assert rewards == [0.0, 0.0, 0.0], "GRPO's advantage is then 0, leaving the OPD term alone"


def _teacher_response(token_ids, log_probs):
    """SGLang's /generate answer with logprob_start_len=0: a placeholder first, then one entry per later input token."""
    return {
        "meta_info": {
            "input_token_logprobs": [[None, token_ids[0], None]]
            + [[lp, tid, None] for lp, tid in zip(log_probs, token_ids[1:], strict=True)]
        }
    }


def _teacher_scored(monkeypatch, answer, collected):
    """Miles' agentic generate returns ``collected``; the teacher answers with ``answer(sample)``."""
    calls = []

    async def fake_generate(input):
        return GenerateFnOutput(samples=collected)

    async def fake_teacher(args, sample):
        calls.append(sample.index)
        return answer(sample)

    async def no_sleep(_):
        pass

    monkeypatch.setattr(generate.agentic_tool_call, "generate", fake_generate)
    monkeypatch.setattr(generate.on_policy_distillation, "reward_func", fake_teacher)
    monkeypatch.setattr(generate.asyncio, "sleep", no_sleep)
    return calls


def _generate(evaluation=False):
    return asyncio.run(generate.generate_with_teacher(SimpleNamespace(args=ARGS, evaluation=evaluation)))


def test_the_sglang_teacher_scores_each_collected_sample(monkeypatch):
    scored = Sample(index=0, tokens=[1, 2, 3, 4, 5], response_length=2, reward=1.0, metadata={"reward": 1.0})
    lost = Sample(index=1, tokens=[1, 2, 3], response_length=1, metadata={"reward": None})
    aborted = Sample(index=2, tokens=[1, 2, 3], response_length=1, metadata={"reward": 0.0})
    aborted.status = Sample.Status.ABORTED
    answer = lambda s: _teacher_response(s.tokens, [-0.1 * t for t in s.tokens[1:]])  # noqa: E731
    calls = _teacher_scored(monkeypatch, answer, [scored, lost, aborted])
    assert _generate().samples == [scored, lost, aborted]
    assert scored.teacher_log_probs == [-0.4, -0.5], "the log-probs of the last response_length tokens"
    assert scored.reward == 1.0, "the task reward stays"
    assert calls == [0], "a sample the sandbox failed to score and an aborted one are not sent"


def test_evaluation_samples_are_not_scored(monkeypatch):
    sample = Sample(index=0, tokens=[1, 2, 3], response_length=1, metadata={"reward": 1.0})
    calls = _teacher_scored(monkeypatch, lambda s: _teacher_response(s.tokens, [-1.0, -1.0]), sample)
    assert _generate(evaluation=True).samples is sample and calls == []


def test_a_teacher_scoring_other_tokens_is_an_error(monkeypatch):
    sample = Sample(index=0, tokens=[1, 2, 3, 4, 5], response_length=2, metadata={"reward": 0.0})
    _teacher_scored(monkeypatch, lambda s: _teacher_response([9, 9, 9, 9, 9], [-1.0] * 4), [sample])
    with pytest.raises(ValueError, match="other tokens"):
        _generate()


def test_a_failing_teacher_is_retried_then_fails_the_rollout(monkeypatch):
    failures = iter([True, False])

    def flaky(sample):
        if next(failures, True):
            raise ConnectionError("teacher restarting")
        return _teacher_response(sample.tokens, [-1.0] * (len(sample.tokens) - 1))

    sample = Sample(index=0, tokens=[1, 2, 3], response_length=1, metadata={"reward": 0.0})
    calls = _teacher_scored(monkeypatch, flaky, [sample])
    _generate()
    assert sample.teacher_log_probs == [-1.0] and len(calls) == 2
    with pytest.raises(RuntimeError, match="failed 3 times"):
        _generate()


def test_the_wrapper_registers_the_agentic_generate_flags():
    """Miles adds a generate function's own flags through its add_arguments; without them train.py rejects
    --custom-agent-function-path (2026-10-23, smoke 01M5MY8EV6WAHDYCSSZYGNF4YN)."""
    assert generate.generate_with_teacher.add_arguments is generate.agentic_tool_call.generate.add_arguments
