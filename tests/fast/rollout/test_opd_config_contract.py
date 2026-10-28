from argparse import Namespace

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.metrics import _compute_passrate_from_samples, _compute_zero_std_metrics
from miles.rollout import on_policy_distillation
from miles.rollout.on_policy_distillation import (
    _compute_topk_reverse_kl,
    _get_opd_top_k,
    _get_reward_weight_mode,
    _get_top_k_strategy,
    post_process_rewards,
    reward_func,
)
from miles.utils.types import Sample


class TestOpdConfigurationContract:
    @pytest.mark.parametrize("value,expected", [(None, 0), (-3, 0), (0, 0), (7, 7)])
    def test_topk_configuration_clamps_nonpositive_values(self, value: int | None, expected: int) -> None:
        """Only a positive configured top-k enables top-k scoring."""
        assert _get_opd_top_k(Namespace(opd_log_prob_top_k=value)) == expected

    def test_unknown_strategy_fails_instead_of_selecting_another_distribution(self) -> None:
        """A misspelled top-k strategy must not silently alter the distillation objective."""
        with pytest.raises(ValueError, match="Unknown OPD top-k strategy"):
            _get_top_k_strategy(Namespace(opd_top_k_strategy="typo"))

    def test_unknown_weight_mode_fails_instead_of_selecting_other_weights(self) -> None:
        """A misspelled weighting mode must not silently alter the distillation objective."""
        with pytest.raises(ValueError, match="Unknown OPD reward weight mode"):
            _get_reward_weight_mode(Namespace(opd_reward_weight_mode="typo"))

    @pytest.mark.parametrize("timeout", [None, 3.25])
    async def test_sampled_token_scoring_uses_selected_teacher_and_configured_timeout(
        self, monkeypatch: pytest.MonkeyPatch, timeout: float | None
    ) -> None:
        """The non-top-k path forwards the selected teacher, input tokens, and optional timeout."""
        calls: list[tuple[str, dict, float | None]] = []
        reply = {"meta_info": {"input_token_logprobs": [None, [-0.4, 2]]}}

        async def fake_post_json(url: str, payload: dict, timeout_secs: float | None = None) -> dict:
            calls.append((url, payload, timeout_secs))
            return reply

        monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post_json)
        args = Namespace(
            opd_log_prob_top_k=0,
            opd_teacher_urls=["math=http://math/generate"],
            opd_teacher_key="task",
            rm_url="http://fallback/generate",
            sglang_router_request_timeout_secs=timeout,
        )

        sample = Sample(tokens=[1, 2], response_length=1, metadata={"task": "math"})
        result = await on_policy_distillation.reward_func(args=args, sample=sample)

        assert result == 0.0
        assert torch.equal(sample.teacher_log_probs, torch.tensor([-0.4]))
        assert len(calls) == 1
        assert calls[0][0] == "http://math/generate"
        assert calls[0][1]["input_ids"] == [1, 2]
        assert calls[0][1]["sampling_params"]["max_new_tokens"] == 0
        assert calls[0][1]["return_logprob"] is True
        assert calls[0][2] == timeout


def _opd_args(top_k: int = 0) -> Namespace:
    return Namespace(
        opd_log_prob_top_k=top_k,
        opd_top_k_strategy="only-student",
        opd_reward_weight_mode="student_p",
        opd_topk_per_position=False,
        opd_teacher_urls=None,
        rm_url="http://teacher/generate",
        sglang_router_request_timeout_secs=None,
        reward_key=None,
    )


def _sampled_token_reply(response_log_probs: list[float]) -> dict:
    """A teacher reply scored from position 0: a placeholder row, two prompt rows, then the response rows."""
    rows = [None, [-9.0, 11, None], [-8.0, 12, None]]
    return {"meta_info": {"input_token_logprobs": rows + [[lp, 13, None] for lp in response_log_probs]}}


class TestOpdRewardKeepsOnlyTheTrainingSignal:
    """The teacher reply is reduced on arrival; the reward slot holds a scalar every rollout metric can read."""

    async def test_sampled_token_reward_is_a_scalar_and_the_sample_carries_the_teacher_log_probs(self, monkeypatch):
        reply = _sampled_token_reply([-0.25, -1.5])

        async def fake_post_json(url, payload, timeout_secs=None):
            return reply

        monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post_json)
        sample = Sample(tokens=[11, 12, 13, 14, 15], response_length=2)

        reward = await reward_func(_opd_args(), sample)

        assert reward == 0.0
        assert torch.equal(sample.teacher_log_probs, torch.tensor([-0.25, -1.5]))
        assert sample.opd_reverse_kl is None

    async def test_topk_reward_is_a_scalar_and_the_sample_carries_the_reverse_kl(self, monkeypatch):
        student_top = [[[-0.1, 5], [-2.0, 6]], [[-0.3, 7], [-1.5, 8]]]
        teacher_rows = [[[-0.2, 5], [-1.0, 6], [-0.4, 7], [-2.5, 8]], [[-0.9, 5], [-3.0, 6], [-0.5, 7], [-1.2, 8]]]
        reply = {"meta_info": {"input_token_ids_logprobs": [None, None, *teacher_rows]}}

        async def fake_post_json(url, payload, timeout_secs=None):
            return reply

        monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post_json)
        sample = Sample(tokens=[1, 2, 3, 4], response_length=2, metadata={"opd_student_top_logprobs": student_top})
        args = _opd_args(top_k=2)

        reward = await reward_func(args, sample)

        assert reward == 0.0
        assert sample.teacher_log_probs is None
        assert torch.equal(sample.opd_reverse_kl, _compute_topk_reverse_kl(args, sample, {"teacher": reply}))

    def test_post_process_keeps_stored_signals_and_still_extracts_raw_teacher_replies(self):
        scored = Sample(tokens=[11, 12, 13, 14], response_length=1, reward=0.0, teacher_log_probs=torch.tensor([-0.7]))
        legacy = Sample(tokens=[11, 12, 13, 14], response_length=1, reward=_sampled_token_reply([-0.6]))

        raw_rewards, rewards = post_process_rewards(_opd_args(), [scored, legacy])

        assert raw_rewards == rewards == [0.0, 0.0]
        assert torch.equal(scored.teacher_log_probs, torch.tensor([-0.7]))
        assert torch.equal(legacy.teacher_log_probs, torch.tensor([-0.6]))

    def test_post_process_refuses_a_sample_with_no_teacher_signal(self):
        with pytest.raises(ValueError, match="neither a teacher reply"):
            post_process_rewards(_opd_args(), [Sample(tokens=[1, 2], response_length=1, reward=0.0)])

    @pytest.mark.parametrize("group_size", [1, 4])
    async def test_scored_rollouts_reach_the_reward_metrics_as_scalars(self, monkeypatch, group_size):
        """With one response per prompt every group is zero-std, so main crashed in round() on the teacher reply."""

        async def fake_post_json(url, payload, timeout_secs=None):
            return _sampled_token_reply([-0.5] * 4)

        monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post_json)
        samples = make_samples_grouped(3, group_size, rewards=[None] * (3 * group_size))
        for sample in samples:
            sample.reward = await reward_func(_opd_args(), sample)
        args = make_args(advantage_estimator="grpo", reward_key=None, n_samples_per_prompt=group_size)

        assert _compute_zero_std_metrics(args, samples)["zero_std/count_0.0"] == 3
        assert _compute_passrate_from_samples(args, samples) == (
            {} if group_size == 1 else {"pass@1": 0.0, "pass@2": 0.0, "pass@4": 0.0}
        )
