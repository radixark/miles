from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    QWEN3_MOE_TRUE_ON_POLICY_V1,
    apply_true_on_policy_script_defaults,
    build_true_on_policy_launch_plan,
    get_megatron_model_type,
    get_true_on_policy_contract,
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
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
        "sglang_expert_parallel_size": 1,
        "true_on_policy_contract": None,
        "use_sequence_parallel": True,
        "num_nodes": 1,
        "num_gpus_per_node": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen3_dense_profile_resolves_model_names():
    profile = get_true_on_policy_model_profile("Qwen3-4B")
    contract = get_true_on_policy_contract("qwen3_dense_true_on_policy_v1")

    assert profile.family == "qwen3_dense"
    assert profile.contract is QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert profile.contract is contract
    assert contract.schema.name == "qwen3_dense_true_on_policy_v1"
    assert contract.schema.model_family == "qwen3_dense"
    assert profile.supported_train_layouts == ("dp", "tp", "pp", "ulysses_cp")
    assert profile.supported_rollout_layouts == ("dp", "tp")
    assert profile.required_kernel_contracts == ("qwen3_dense_sglang_math",)
    assert profile.logprob_contract == "sglang_prefill"
    assert profile.supports_ulysses_cp
    assert profile.supports_tp_invariant
    assert get_megatron_model_type("Qwen3-4B") == "qwen3-4B"
    assert get_megatron_model_type("Qwen3-4B-Instruct-2507") == "qwen3-4B-Instruct-2507"


def test_unknown_true_on_policy_model_fails_early():
    with pytest.raises(ValueError, match="does not have a model profile"):
        get_true_on_policy_model_profile("unknown-model")


def test_qwen3_moe_profile_resolves_ep_contract():
    profile = get_true_on_policy_model_profile("Qwen3-30B-A3B")
    contract = get_true_on_policy_contract("qwen3_moe_true_on_policy_v1")

    assert profile.family == "qwen3_moe"
    assert profile.contract is QWEN3_MOE_TRUE_ON_POLICY_V1
    assert profile.contract is contract
    assert contract.schema.model_family == "qwen3_moe"
    assert profile.required_kernel_contracts == (
        "qwen3_dense_sglang_math",
        "qwen3_moe_sglang_math",
    )
    assert "ep" in profile.supported_train_layouts
    assert profile.supports_ep_invariant
    assert get_megatron_model_type("Qwen3-30B-A3B") == "qwen3-30B-A3B"


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

    assert plan.sglang_target == expected_target
    assert plan.contract is QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert plan.sglang_args.values == (
        "--sglang-enable-deterministic-inference",
        "--sglang-true-on-policy-contract",
        "qwen3_dense_true_on_policy_v1",
        "--sglang-attention-backend",
        "fa3",
    )


def test_qwen3_moe_ep_is_separate_from_tp_invariant_rollout():
    args = _args(
        model_name="Qwen3-30B-A3B",
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus_per_engine=1,
        true_on_policy_contract=None,
    )

    plan = build_true_on_policy_launch_plan(args)

    assert plan.sglang_target == "fsdp"
    assert plan.contract is QWEN3_MOE_TRUE_ON_POLICY_V1
    assert plan.parallel_layout is not None
    assert plan.parallel_layout.uses_train_ep
    assert not plan.parallel_layout.uses_train_tp
    assert not plan.parallel_layout.uses_train_expert_tp
    assert plan.kernel_policy is not None
    assert not plan.kernel_policy.tp_invariant_row_linear
    assert not plan.kernel_policy.deterministic_tp_allreduce
    assert plan.kernel_policy.ep_invariant_moe
    assert plan.kernel_policy.deterministic_moe_routing
    assert plan.kernel_policy.moe_topk_tiebreak == "stable_sort"
    assert not plan.kernel_policy.disable_sglang_cuda_graph
    assert plan.env_vars["MILES_TRUE_ON_POLICY_ROLLOUT_TP_SIZE"] == "1"
    assert plan.env_vars["SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM"] == "1"
    assert plan.env_vars["MODEL_ARGS_DISABLE_MOE_PERMUTE_FUSION"] == "1"
    assert "--sglang-disable-cuda-graph" not in plan.train_args
    assert "--sglang-true-on-policy-contract qwen3_moe_true_on_policy_v1" in plan.train_args
    assert "--true-on-policy-contract qwen3_moe_true_on_policy_v1" in plan.train_args


def test_qwen3_moe_rollout_ep_does_not_enable_unverified_dp_attention():
    args = _args(
        model_name="Qwen3-30B-A3B",
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus_per_engine=4,
        sglang_expert_parallel_size=4,
    )

    plan = build_true_on_policy_launch_plan(args)

    assert plan.kernel_policy is not None
    assert plan.kernel_policy.sglang_attention_data_parallel_size == 1
    assert "--sglang-data-parallel-size" not in plan.train_args
    assert "--sglang-enable-dp-attention" not in plan.train_args


def test_qwen3_moe_rollout_tp_still_enables_tp_invariant_math():
    args = _args(
        model_name="Qwen3-30B-A3B",
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        expert_model_parallel_size=4,
        rollout_num_gpus_per_engine=8,
    )

    plan = build_true_on_policy_launch_plan(args)

    assert plan.sglang_target == "fsdp_tp"
    assert plan.kernel_policy is not None
    assert plan.kernel_policy.tp_invariant_row_linear
    assert plan.kernel_policy.deterministic_tp_allreduce
    assert plan.kernel_policy.ep_invariant_moe
    assert plan.env_vars["MILES_TRUE_ON_POLICY_ROLLOUT_TP_SIZE"] == "8"
    assert plan.env_vars["SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM"] == "1"


@pytest.mark.parametrize(
    ("tp_size", "ep_size", "cp_size"),
    [
        (1, 4, 2),
        (4, 1, 2),
    ],
)
def test_qwen3_moe_valid_8_gpu_megatron_training_topologies(
    tp_size: int,
    ep_size: int,
    cp_size: int,
):
    args = _args(
        model_name="Qwen3-30B-A3B",
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        rollout_num_gpus_per_engine=max(tp_size, ep_size),
        sglang_expert_parallel_size=ep_size,
        num_nodes=1,
        num_gpus_per_node=8,
    )

    plan = build_true_on_policy_launch_plan(args)

    assert plan.parallel_layout is not None
    assert plan.parallel_layout.train_tensor_parallel_size == tp_size
    assert plan.parallel_layout.train_expert_model_parallel_size == ep_size
    assert plan.parallel_layout.train_context_parallel_size == cp_size


def test_qwen3_moe_rejects_train_tp_with_ep_until_sequence_parallel_is_supported():
    args = _args(
        model_name="Qwen3-30B-A3B",
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus_per_engine=4,
        sglang_expert_parallel_size=4,
        num_nodes=1,
        num_gpus_per_node=8,
    )

    with pytest.raises(ValueError, match="does not support train TP with EP yet"):
        build_true_on_policy_launch_plan(args)


def test_qwen3_moe_rejects_ep_that_does_not_fit_8_gpu_megatron_dp():
    args = _args(
        model_name="Qwen3-30B-A3B",
        tensor_model_parallel_size=1,
        context_parallel_size=4,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus_per_engine=4,
        sglang_expert_parallel_size=4,
        num_nodes=1,
        num_gpus_per_node=8,
    )

    with pytest.raises(ValueError, match="needs at least 16 train GPUs"):
        build_true_on_policy_launch_plan(args)


def test_contract_object_owns_miles_kernel_policy_values():
    args = _args(
        train_backend="megatron",
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
    )

    plan = build_true_on_policy_launch_plan(args)

    assert plan.kernel_policy is not None
    assert plan.kernel_policy.contract is QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert plan.kernel_policy.sglang_attention_backend == "fa3"
    assert plan.kernel_policy.megatron_uses_sglang_backend
    assert plan.kernel_policy.disable_rope_fusion
    assert plan.kernel_policy.disable_bias_swiglu_fusion
    assert plan.kernel_policy.batch_invariant_mode
    assert plan.kernel_policy.tp_invariant_row_linear
    assert plan.kernel_policy.deterministic_tp_allreduce


def test_megatron_true_on_policy_disables_sequence_parallel_and_enables_backend_flags():
    args = _args(train_backend="megatron", use_sequence_parallel=True)

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is False
    assert "--use-sglang" not in plan.train_args
    assert "--true-on-policy-contract qwen3_dense_true_on_policy_v1" in plan.train_args
    assert "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1" in plan.train_args
    assert "--recompute-logprobs-via-prefill" not in plan.train_args
    assert "--batch-invariant-mode" in plan.train_args
    assert "--no-rope-fusion" in plan.train_args
    assert "ROW_LINEAR_ENABLE_INV" not in plan.env_vars
    assert "MEGATRON_USE_DETERMINISTIC_ALLREDUCE" not in plan.env_vars


def test_megatron_tp2_cp4_normal_topology_has_complete_true_on_policy_contract(monkeypatch):
    monkeypatch.delenv("NCCL_ALGO", raising=False)

    args = _args(
        train_backend="megatron",
        tensor_model_parallel_size=2,
        context_parallel_size=4,
        pipeline_model_parallel_size=1,
        rollout_num_gpus_per_engine=8,
        use_sequence_parallel=True,
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is False
    assert plan.parallel_layout is not None
    assert plan.parallel_layout.train_tensor_parallel_size == 2
    assert plan.parallel_layout.train_context_parallel_size == 4
    assert plan.parallel_layout.rollout_num_gpus_per_engine == 8
    assert plan.parallel_layout.uses_train_tp
    assert plan.parallel_layout.uses_ulysses_cp
    assert plan.parallel_layout.uses_rollout_tp
    assert plan.kernel_policy is not None
    assert plan.kernel_policy.tp_invariant_row_linear
    assert plan.kernel_policy.deterministic_tp_allreduce
    assert plan.sglang_args.values == (
        "--sglang-enable-deterministic-inference",
        "--sglang-true-on-policy-contract",
        "qwen3_dense_true_on_policy_v1",
        "--sglang-attention-backend",
        "fa3",
    )
    assert plan.megatron_args.values == (
        "--true-on-policy-contract",
        "qwen3_dense_true_on_policy_v1",
        "--transformer-impl",
        "local",
        "--use-cpu-initialization",
        "--batch-invariant-mode",
        "--no-bias-swiglu-fusion",
        "--no-rope-fusion",
    )
    assert plan.miles_args.values == (
        "--deterministic-mode",
        "--true-on-policy-mode",
    )
    assert plan.env_vars == {
        "NCCL_ALGO": "Ring",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MILES_TRUE_ON_POLICY_ROLLOUT_TP_SIZE": "8",
        "SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM": "1",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "0",
    }


def test_true_on_policy_contract_override_is_validated():
    args = _args(true_on_policy_contract="unknown_contract")

    with pytest.raises(ValueError, match="Unsupported true-on-policy contract"):
        build_true_on_policy_launch_plan(args)


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
    assert plan.fsdp_args.values == ("--attn-implementation", "flash_attention_3")
    assert plan.megatron_args.values == ()
    assert "--attn-implementation flash_attention_3" in plan.train_args
    assert "--use-sglang" not in plan.train_args
    assert "ROW_LINEAR_ENABLE_INV" not in plan.env_vars


def test_off_policy_builds_empty_launch_plan_and_does_not_mutate_args():
    args = _args(true_on_policy=False, use_sequence_parallel=True)

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is True
    assert not plan.enabled
    assert plan.train_args == ""
    assert plan.env_vars == {}
