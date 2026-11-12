from argparse import Namespace

import pytest

from miles.utils.rollout_topk_logprobs import candidates_reference_sampling_mask, validate_rollout_topk_logprobs_args


def _args(**overrides: object) -> Namespace:
    values = dict(
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        rollout_top_p=1.0,
        rollout_top_k=-1,
        use_opd=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
    )
    return Namespace(**(values | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"rollout_top_logprobs_num": 128},
        {"rollout_top_logprobs_num": 128, "rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
        {
            "rollout_top_logprobs_num": 64,
            "rollout_sampling_logprobs_mode": "support",
            "rollout_top_p": 0.9,
            "rollout_top_k": 64,
        },
        {"rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
        {"rollout_sampling_logprobs_mode": "support", "rollout_top_p": 0.9, "rollout_top_k": 64},
        {
            "rollout_top_logprobs_num": 8,
            "use_opd": True,
            "opd_log_prob_top_k": 4,
            "opd_top_k_strategy": "only-teacher",
        },
    ],
)
def test_consistent_settings_pass(overrides: dict) -> None:
    validate_rollout_topk_logprobs_args(_args(**overrides))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rollout_top_logprobs_num": -1}, "non-negative"),
        (
            {"rollout_top_logprobs_num": 32, "rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
            ">= --rollout-top-k",
        ),
        ({"rollout_top_logprobs_num": 8, "use_opd": True, "opd_log_prob_top_k": 4}, "OPD student"),
    ],
)
def test_inconsistent_settings_are_rejected(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_rollout_topk_logprobs_args(_args(**overrides))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"loss_type": "score_centering"}, False),
        ({"loss_type": "score_centering", "rollout_top_logprobs_num": 64}, False),
        (
            {
                "loss_type": "score_centering",
                "rollout_top_logprobs_num": 64,
                "rollout_sampling_logprobs_mode": "support",
            },
            True,
        ),
        (
            {"loss_type": "policy_loss", "rollout_top_logprobs_num": 64, "rollout_sampling_logprobs_mode": "support"},
            False,
        ),
    ],
)
def test_score_centering_support_candidates_reference_the_sampling_mask(overrides: dict, expected: bool) -> None:
    assert candidates_reference_sampling_mask(_args(**overrides)) is expected
