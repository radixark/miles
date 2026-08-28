import pytest

from run import ScriptArgs, _grpo_args


def test_grpo_args_uses_configured_kl_loss_coefficient() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        kl_loss_coef=0.01,
    )

    assert "--kl-loss-coef 0.01 " in _grpo_args(args)


def test_script_args_rejects_negative_kl_loss_coefficient() -> None:
    with pytest.raises(ValueError, match="kl_loss_coef must be nonnegative"):
        ScriptArgs(
            hardware="H200",
            num_gpus_per_node=8,
            kl_loss_coef=-0.01,
        )
