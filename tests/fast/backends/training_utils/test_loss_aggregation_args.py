import argparse
from types import SimpleNamespace

import pytest


def _args(**overrides):
    values = dict(
        loss_aggregation="sample_mean",
        loss_aggregation_divisor=None,
        calculate_per_token_loss=False,
        custom_tis_function_path=None,
        indep_dp=False,
        loss_type="policy_loss",
        entropy_coef=0.0,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        global_batch_size=4,
        n_samples_per_prompt=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("divisor", [None, 0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_constant_requires_positive_finite_divisor(miles, divisor):
    with pytest.raises(ValueError, match="loss-aggregation-divisor"):
        miles.arguments._validate_loss_aggregation_args(
            _args(loss_aggregation="constant", loss_aggregation_divisor=divisor)
        )


def test_unknown_mode_from_custom_config_is_rejected(miles):
    with pytest.raises(ValueError, match="Unknown --loss-aggregation mode"):
        miles.arguments._validate_loss_aggregation_args(_args(loss_aggregation="typo"))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"loss_aggregation": ["token_mean"]}, "must be a string"),
        ({"calculate_per_token_loss": "false"}, "must be a boolean"),
    ],
)
def test_loss_aggregation_config_types_are_strict(miles, overrides, match):
    with pytest.raises(ValueError, match=match):
        miles.arguments._validate_loss_aggregation_args(_args(**overrides))


@pytest.mark.parametrize("mode", ["prompt_mean", "token_mean"])
def test_custom_tis_is_rejected_when_step_normalizer_can_change(miles, mode):
    with pytest.raises(ValueError, match="custom-tis-function-path"):
        miles.arguments._validate_loss_aggregation_args(
            _args(loss_aggregation=mode, custom_tis_function_path="pkg.module:custom_tis")
        )


@pytest.mark.parametrize("mode", ["sample_mean", "constant"])
def test_custom_tis_is_allowed_with_stable_step_normalizers(miles, mode):
    miles.arguments._validate_loss_aggregation_args(
        _args(
            loss_aggregation=mode,
            loss_aggregation_divisor=10.0 if mode == "constant" else None,
            custom_tis_function_path="pkg.module:custom_tis",
        )
    )


@pytest.mark.parametrize(
    ("overrides", "expected_mode", "expected_per_token"),
    [
        ({}, "sample_mean", False),
        ({"loss_aggregation": "token_mean"}, "token_mean", True),
        ({"calculate_per_token_loss": True}, "token_mean", True),
        ({"loss_aggregation": "constant", "loss_aggregation_divisor": 10.0}, "constant", False),
    ],
)
def test_aliases_resolve_to_one_loss_aggregation_axis(miles, overrides, expected_mode, expected_per_token):
    args = _args(**overrides)

    miles.arguments._validate_loss_aggregation_args(args)

    assert args.loss_aggregation == expected_mode
    assert args.calculate_per_token_loss is expected_per_token


@pytest.mark.parametrize("mode", ["constant", "prompt_mean"])
def test_non_token_mode_rejects_calculate_per_token_loss(miles, mode):
    args = _args(
        loss_aggregation=mode,
        loss_aggregation_divisor=10.0 if mode == "constant" else None,
        calculate_per_token_loss=True,
    )

    with pytest.raises(ValueError, match="incompatible with --calculate-per-token-loss"):
        miles.arguments._validate_loss_aggregation_args(args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"entropy_coef": 0.01},
        {"use_kl_loss": True, "kl_loss_coef": 0.1},
    ],
)
def test_token_mean_rejects_sample_normalized_auxiliary_losses(miles, overrides):
    with pytest.raises(ValueError, match="auxiliary policy-loss terms are sample-normalized"):
        miles.arguments._validate_loss_aggregation_args(_args(loss_aggregation="token_mean", **overrides))


def test_token_mean_allows_zero_weight_auxiliary_metrics(miles):
    miles.arguments._validate_loss_aggregation_args(
        _args(loss_aggregation="token_mean", use_kl_loss=True, kl_loss_coef=0.0)
    )


@pytest.mark.parametrize(
    ("mode", "global_batch_size", "should_pass"),
    [
        ("prompt_mean", 3, False),
        ("prompt_mean", 6, True),
        ("sample_mean", 3, True),
    ],
)
def test_prompt_mean_requires_whole_groups_per_step(miles, mode, global_batch_size, should_pass):
    args = _args(loss_aggregation=mode, global_batch_size=global_batch_size)
    if should_pass:
        miles.arguments._validate_loss_aggregation_args(args)
    else:
        with pytest.raises(ValueError, match="multiple of n_samples_per_prompt"):
            miles.arguments._validate_loss_aggregation_args(args)


_BASE_ARGV = [
    "--rollout-batch-size",
    "4",
    "--n-samples-per-prompt",
    "2",
    "--num-steps-per-rollout",
    "2",
    "--num-rollout",
    "1",
]


def _parse_and_validate(miles, *extra_args, config_text=None, tmp_path=None):
    parser = argparse.ArgumentParser()
    miles.arguments.get_miles_extra_args_provider()(parser)
    argv = [*_BASE_ARGV, *extra_args]
    if config_text is not None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_text)
        argv.extend(["--custom-config-path", str(config_path)])
    args = parser.parse_args(argv)
    miles.arguments.miles_validate_args(args)
    return args


def test_validation_uses_derived_global_batch_size(miles):
    argv = [
        "--rollout-batch-size",
        "3",
        "--n-samples-per-prompt",
        "2",
        "--num-steps-per-rollout",
        "2",
        "--num-rollout",
        "1",
        "--loss-aggregation",
        "prompt_mean",
    ]
    parser = argparse.ArgumentParser()
    miles.arguments.get_miles_extra_args_provider()(parser)

    with pytest.raises(ValueError, match="multiple of n_samples_per_prompt"):
        miles.arguments.miles_validate_args(parser.parse_args(argv))


def test_validation_runs_after_custom_config_overrides(miles, tmp_path):
    config = "loss_aggregation: prompt_mean\nrollout_batch_size: 3\n"

    with pytest.raises(ValueError, match="multiple of n_samples_per_prompt"):
        _parse_and_validate(miles, config_text=config, tmp_path=tmp_path)


def test_custom_global_batch_size_must_match_derived_value(miles, tmp_path):
    config = "rollout_batch_size: 3\nglobal_batch_size: 4\n"

    with pytest.raises(ValueError, match="global_batch_size 4 is not equal"):
        _parse_and_validate(miles, config_text=config, tmp_path=tmp_path)


@pytest.mark.parametrize(
    "config",
    [
        "loss_aggregation: sample_mean\ncalculate_per_token_loss: true\n",
        "loss_aggregation: token_mean\ncalculate_per_token_loss: false\n",
    ],
)
def test_custom_config_rejects_conflicting_alias_values(miles, tmp_path, config):
    with pytest.raises(ValueError, match="conflicting values"):
        _parse_and_validate(miles, config_text=config, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("cli_args", "config_mode", "expected_mode", "expected_per_token"),
    [
        ((), "token_mean", "token_mean", True),
        (("--loss-aggregation", "token_mean"), "sample_mean", "sample_mean", False),
        (("--calculate-per-token-loss",), "sample_mean", "sample_mean", False),
    ],
)
def test_custom_config_overrides_both_cli_spellings(
    miles,
    tmp_path,
    cli_args,
    config_mode,
    expected_mode,
    expected_per_token,
):
    args = _parse_and_validate(
        miles,
        *cli_args,
        config_text=f"loss_aggregation: {config_mode}\n",
        tmp_path=tmp_path,
    )

    assert args.loss_aggregation == expected_mode
    assert args.calculate_per_token_loss is expected_per_token
