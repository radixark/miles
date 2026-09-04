import pytest

from run import (
    ScriptArgs,
    _checkpoint_args,
    _grpo_args,
    _misc_args,
    _optimizer_args,
    _prompt_rows,
    _rollout_args,
    _sglang_args,
)


def test_prompt_rows_pass_system_prompt_selection_to_chess_harness() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        rollout_batch_size=2,
        system_prompt_variant="random",
    )

    rows = _prompt_rows(args)

    assert [row["metadata"]["chess"]["system_prompt_variant"] for row in rows] == [
        "random",
        "random",
    ]


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


def test_grpo_args_uses_configured_repetition_reward_penalty() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        repetition_reward_penalty=0.1,
    )

    assert "--repetition-reward-penalty 0.1 " in _grpo_args(args)


def test_script_args_rejects_negative_repetition_reward_penalty() -> None:
    with pytest.raises(
        ValueError,
        match="repetition_reward_penalty must be nonnegative",
    ):
        ScriptArgs(
            hardware="H200",
            num_gpus_per_node=8,
            repetition_reward_penalty=-0.1,
        )


def test_optimizer_args_uses_configured_learning_rate() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        learning_rate=3e-7,
    )

    assert "--lr 3e-07 " in _optimizer_args(args)


def test_script_args_rejects_nonpositive_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        ScriptArgs(
            hardware="H200",
            num_gpus_per_node=8,
            learning_rate=0.0,
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


def test_fully_async_requires_disaggregated_nodes() -> None:
    with pytest.raises(ValueError, match="fully_async requires at least two nodes"):
        ScriptArgs(
            hardware="H200",
            num_gpus_per_node=8,
            num_nodes=1,
            fully_async=True,
        )


def test_fully_async_uses_continuous_disaggregated_rollout() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        num_nodes=2,
        train_num_nodes=1,
        fully_async=True,
    )

    rollout_args = _rollout_args(args)
    misc_args = _misc_args(args)

    assert "--fully-async " in rollout_args
    assert "--pause-generation-mode in_place " in rollout_args
    assert "--use-tis " in _grpo_args(args)
    assert "--actor-num-nodes 1 " in misc_args
    assert "--rollout-num-gpus 8 " in misc_args
    assert "--colocate " not in misc_args


def test_synchronous_mode_remains_colocated() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        num_nodes=1,
    )

    assert "--fully-async " not in _rollout_args(args)
    assert "--use-tis " not in _grpo_args(args)
    assert "--colocate " in _misc_args(args)


def test_qwen38_dense_defaults_match_supported_recipe() -> None:
    args = ScriptArgs(hardware="H200", num_gpus_per_node=8)

    assert args.model_name == "Qwen3.8-27B"
    assert args.megatron_model_type == "qwen3.8-27B"
    assert (args.tp, args.pp, args.cp, args.ep, args.etp) == (4, 1, 1, 1, 1)
    assert args.rollout_num_gpus_per_engine == 1
    assert args.sglang_mem_fraction_static == 0.8


def test_qwen38_dense_rollout_omits_moe_and_speculative_flags() -> None:
    args = ScriptArgs(hardware="H200", num_gpus_per_node=8)
    sglang_args = _sglang_args(args)

    assert "--rollout-num-gpus-per-engine 1 " in sglang_args
    assert "--sglang-speculative-algorithm" not in sglang_args
    assert "--sglang-ep-size" not in sglang_args
    assert "--moe-token-dispatcher-type" not in _misc_args(args)
