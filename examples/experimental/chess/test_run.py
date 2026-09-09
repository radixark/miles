import pytest

from run import (
    ScriptArgs,
    _agent_args,
    _checkpoint_args,
    _extra_env_vars,
    _grpo_args,
    _misc_args,
    _optimizer_args,
    _prompt_rows,
    _rollout_args,
    _sglang_args,
)


def test_extra_env_vars_forward_nonempty_ld_library_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib:/opt/nccl/lib")
    args = ScriptArgs(hardware="H200", num_gpus_per_node=8)

    assert _extra_env_vars(args)["LD_LIBRARY_PATH"] == "/opt/cuda/lib:/opt/nccl/lib"


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
    assert all(row["metadata"]["chess"]["max_llm_retries_per_move"] == 0 for row in rows)
    assert "--custom-rollout-log-function-path chess_training.log_rollout_metrics " in _agent_args(args)


@pytest.mark.parametrize("retries", [3, 5])
def test_prompt_rows_forward_configured_retry_budget(retries: int) -> None:
    args = ScriptArgs(hardware="H200", num_gpus_per_node=8, max_llm_retries_per_move=retries)
    assert _prompt_rows(args)[0]["metadata"]["chess"]["max_llm_retries_per_move"] == retries


@pytest.mark.parametrize("retries", [-1, True, 1.5])
def test_prompt_rows_reject_invalid_retry_budget(retries: object) -> None:
    with pytest.raises(ValueError, match="max_llm_retries_per_move"):
        ScriptArgs(hardware="H200", num_gpus_per_node=8, max_llm_retries_per_move=retries)


def test_grpo_args_uses_configured_kl_loss_coefficient() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        kl_loss_coef=0.01,
    )

    assert "--kl-loss-coef 0.01 " in _grpo_args(args)


@pytest.mark.parametrize("kl_loss_type", ["low_var_kl", "k3"])
def test_grpo_args_forward_kl_estimator(kl_loss_type: str) -> None:
    args = ScriptArgs(hardware="H200", num_gpus_per_node=8, kl_loss_type=kl_loss_type)
    grpo_args = _grpo_args(args)
    assert f"--kl-loss-type {kl_loss_type} " in grpo_args
    assert grpo_args.count("--kl-loss-type ") == 1


def test_invalid_kl_estimator_is_rejected() -> None:
    with pytest.raises(ValueError, match="kl_loss_type"):
        ScriptArgs(hardware="H200", num_gpus_per_node=8, kl_loss_type="unknown")


@pytest.mark.parametrize("harness_mode", ["conversation", "stateful"])
def test_prompt_rows_forward_harness_mode(harness_mode: str) -> None:
    args = ScriptArgs(
        hardware="H200", num_gpus_per_node=8, rollout_batch_size=3, harness_mode=harness_mode
    )
    assert all(row["metadata"]["chess"]["harness_mode"] == harness_mode for row in _prompt_rows(args))


def test_harness_and_kl_defaults_preserve_existing_behavior() -> None:
    args = ScriptArgs(hardware="H200", num_gpus_per_node=8)
    assert args.harness_mode == "conversation"
    assert args.kl_loss_type == "low_var_kl"


def test_invalid_harness_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="harness_mode"):
        ScriptArgs(hardware="H200", num_gpus_per_node=8, harness_mode="unknown")


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

    assert "--repetition-reward-penalty 0 " in _grpo_args(args)
    assert _prompt_rows(args)[0]["metadata"]["chess"]["repetition_reward_penalty"] == 0.1
    assert "--session-sample-postprocessor-path chess_training.postprocess_samples " in _agent_args(args)


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
    assert "--tito-model qwen38small " in _agent_args(args)
    assert (args.tp, args.pp, args.cp, args.ep, args.etp) == (4, 1, 1, 1, 1)
    assert args.rollout_num_gpus_per_engine == 1
    assert args.sglang_mem_fraction_static == 0.8


def test_qwen36_uses_selected_native_tito_family_and_zero_retry_policy() -> None:
    args = ScriptArgs(
        hardware="H200",
        num_gpus_per_node=8,
        model_name="Qwen3.6-35B-A3B",
        megatron_model_type="qwen3.6-35B-A3B",
        tito_model="qwen36",
        model_dir="/models",
        rollout_max_response_len=16384,
        repetition_reward_penalty=0.5,
        max_llm_retries_per_move=0,
        system_prompt_variant="random",
    )
    assert "--tito-model qwen36 " in _agent_args(args)
    assert "qwen38small" not in _agent_args(args)
    assert "--ref-load /models/Qwen3.6-35B-A3B_torch_dist " in _checkpoint_args(args)
    assert "--rollout-max-response-len 16384 " in _rollout_args(args)
    assert "--fully-async " not in _rollout_args(args)
    assert "--colocate " in _misc_args(args)
    chess = _prompt_rows(args)[0]["metadata"]["chess"]
    assert chess["max_llm_retries_per_move"] == 0
    assert chess["repetition_reward_penalty"] == 0.5
    assert chess["system_prompt_variant"] == "random"


def test_qwen38_dense_rollout_omits_moe_and_speculative_flags() -> None:
    args = ScriptArgs(hardware="H200", num_gpus_per_node=8)
    sglang_args = _sglang_args(args)

    assert "--rollout-num-gpus-per-engine 1 " in sglang_args
    assert "--sglang-speculative-algorithm" not in sglang_args
    assert "--sglang-ep-size" not in sglang_args
    assert "--moe-token-dispatcher-type" not in _misc_args(args)
