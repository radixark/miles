import pytest

from run import ScriptArgs, _checkpoint_args, _grpo_args


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


def test_checkpoint_args_can_override_scheduler_when_resuming() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        load_checkpoint_path="/checkpoints/chess",
        override_opt_param_scheduler=True,
    )

    checkpoint_args = _checkpoint_args(args)

    assert "--load /checkpoints/chess " in checkpoint_args
    assert "--override-opt_param-scheduler " in checkpoint_args


def test_scheduler_override_requires_resume_checkpoint() -> None:
    with pytest.raises(
        ValueError,
        match="override_opt_param_scheduler requires load_checkpoint_path",
    ):
        ScriptArgs(
            hardware="H200",
            num_gpus_per_node=8,
            override_opt_param_scheduler=True,
        )
