from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.true_on_policy import (
    apply_true_on_policy_script_defaults,
    build_true_on_policy_launch_plan,
    get_megatron_model_type,
    get_true_on_policy_model_profile,
)


def _args(**overrides):
    values = {
        "true_on_policy": True,
        "model_name": "Qwen3-4B",
        "train_backend": "megatron",
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
        "sglang_rl_on_policy_target": None,
        "use_sequence_parallel": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen3_dense_profile_resolves_model_names():
    profile = get_true_on_policy_model_profile("Qwen3-4B")

    assert profile.family == "qwen3_dense"
    assert profile.supports_ulysses_cp
    assert profile.supports_tp_invariant
    assert get_megatron_model_type("Qwen3-4B") == "qwen3-4B"
    assert get_megatron_model_type("Qwen3-4B-Instruct-2507") == "qwen3-4B-Instruct-2507"


def test_unknown_true_on_policy_model_fails_early():
    with pytest.raises(ValueError, match="does not have a model profile"):
        get_true_on_policy_model_profile("unknown-model")


@pytest.mark.parametrize(
    ("tp_size", "rollout_tp_size", "expected_target"),
    [
        (1, 1, "fsdp"),
        (2, 1, "fsdp_tp"),
        (1, 2, "fsdp_tp"),
    ],
)
def test_true_on_policy_target_is_derived_from_train_and_rollout_tp(
    tp_size: int,
    rollout_tp_size: int,
    expected_target: str,
):
    args = _args(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=rollout_tp_size,
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.sglang_rl_on_policy_target == expected_target
    assert plan.sglang_target == expected_target
    assert f"--sglang-rl-on-policy-target {expected_target}" in plan.train_args


def test_true_on_policy_override_is_preserved_for_expert_debugging():
    args = _args(
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
        sglang_rl_on_policy_target="fsdp",
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.sglang_rl_on_policy_target == "fsdp"
    assert plan.sglang_target == "fsdp"
    assert "ROW_LINEAR_ENABLE_INV" not in plan.env_vars


def test_megatron_true_on_policy_disables_sequence_parallel_and_enables_backend_flags():
    args = _args(train_backend="megatron", use_sequence_parallel=True)

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is False
    assert "--use-sglang" in plan.train_args
    assert "--batch-invariant-mode" in plan.train_args
    assert "--no-rope-fusion" in plan.train_args
    assert plan.env_vars["ROW_LINEAR_ENABLE_INV"] == "1"
    assert plan.env_vars["MEGATRON_USE_DETERMINISTIC_ALLREDUCE"] == "1"


def test_fsdp_true_on_policy_uses_fsdp_attention_without_megatron_backend_flags():
    args = _args(
        train_backend="fsdp",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is True
    assert plan.sglang_target == "fsdp"
    assert "--attn-implementation flash_attention_3" in plan.train_args
    assert "--use-sglang" not in plan.train_args
    assert "ROW_LINEAR_ENABLE_INV" not in plan.env_vars


def test_off_policy_builds_empty_launch_plan_and_does_not_mutate_args():
    args = _args(true_on_policy=False, use_sequence_parallel=True)

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is True
    assert args.sglang_rl_on_policy_target is None
    assert not plan.enabled
    assert plan.train_args == ""
    assert plan.env_vars == {}
