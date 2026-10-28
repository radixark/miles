from argparse import Namespace

import pytest

from miles.rollout import on_policy_distillation
from miles.rollout.on_policy_distillation import _get_opd_top_k, _get_reward_weight_mode, _get_top_k_strategy
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
        reply = {"meta_info": {"input_token_logprobs": [[-0.4, 2]]}}

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

        result = await on_policy_distillation.reward_func(
            args=args, sample=Sample(tokens=[1, 2], response_length=1, metadata={"task": "math"})
        )

        assert result is reply
        assert len(calls) == 1
        assert calls[0][0] == "http://math/generate"
        assert calls[0][1]["input_ids"] == [1, 2]
        assert calls[0][1]["sampling_params"]["max_new_tokens"] == 0
        assert calls[0][1]["return_logprob"] is True
        assert calls[0][2] == timeout
