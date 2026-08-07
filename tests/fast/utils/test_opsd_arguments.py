import argparse

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args

register_cpu_ci(est_time=30, suite="stage-a-cpu")

REQUIRED_ARGS = ["--rollout-batch-size", "64"]


class TestOPSDArguments:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    def test_parses_opinionated_defaults(self):
        args = self._parse([])

        assert args.opsd_type is None
        assert args.opsd_teacher_top_k == 64
        assert args.opsd_pointwise_kl_clip == 0.05
        assert args.opsd_teacher_prompt_function_path is None
        assert args.opsd_teacher_chat_template_kwargs == {}

    def test_sglang_opsd_accepts_complete_pure_configuration(self):
        args = self._parse(
            [
                "--loss-type",
                "opsd_loss",
                "--disable-compute-advantages-and-returns",
                "--opsd-type",
                "sglang",
                "--opsd-teacher-url",
                "http://127.0.0.1:13141/generate",
                "--label-key",
                "label",
            ]
        )

        miles_validate_args(args)

        assert args.loss_type == "opsd_loss"
        assert args.compute_advantages_and_returns is False

    def test_rejects_opsd_with_advantage_computation(self):
        args = self._parse(
            [
                "--loss-type",
                "opsd_loss",
                "--opsd-type",
                "sglang",
                "--opsd-teacher-url",
                "http://127.0.0.1:13141/generate",
                "--label-key",
                "label",
            ]
        )

        with pytest.raises(ValueError, match="--disable-compute-advantages-and-returns"):
            miles_validate_args(args)

    @pytest.mark.parametrize(
        "hybrid_args",
        [
            ["--kl-coef", "0.1"],
            ["--use-kl-loss"],
            ["--entropy-coef", "0.1"],
        ],
    )
    def test_rejects_additive_rl_objective_terms(self, hybrid_args):
        args = self._parse(
            [
                "--loss-type",
                "opsd_loss",
                "--disable-compute-advantages-and-returns",
                "--opsd-type",
                "sglang",
                "--opsd-teacher-url",
                "http://127.0.0.1:13141/generate",
                "--label-key",
                "label",
                *hybrid_args,
            ]
        )

        with pytest.raises(ValueError, match="pure objective"):
            miles_validate_args(args)

    @pytest.mark.parametrize("top_k", ["0", "1"])
    def test_rejects_degenerate_teacher_support(self, top_k):
        args = self._parse(
            [
                "--loss-type",
                "opsd_loss",
                "--disable-compute-advantages-and-returns",
                "--opsd-type",
                "sglang",
                "--opsd-teacher-url",
                "http://127.0.0.1:13141/generate",
                "--opsd-teacher-top-k",
                top_k,
                "--label-key",
                "label",
            ]
        )

        with pytest.raises(ValueError, match="at least 2"):
            miles_validate_args(args)

    def test_rejects_non_object_teacher_chat_template_kwargs(self):
        args = self._parse(
            [
                "--loss-type",
                "opsd_loss",
                "--disable-compute-advantages-and-returns",
                "--opsd-type",
                "sglang",
                "--opsd-teacher-url",
                "http://127.0.0.1:13141/generate",
                "--opsd-teacher-chat-template-kwargs",
                "[]",
                "--label-key",
                "label",
            ]
        )

        with pytest.raises(ValueError, match="JSON object"):
            miles_validate_args(args)
