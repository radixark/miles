from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.true_on_policy import (
    TRUE_ON_POLICY_V1,
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
        "rollout_num_gpus_per_engine": 1,
        "sglang_rl_on_policy_target": None,
        "true_on_policy_contract": None,
        "use_sequence_parallel": True,
        # cp>1 now requires an explicit Ulysses declaration; megatron's default is p2p (ring),
        # which the contract refuses.
        "cp_comm_type": "a2a",
        "allgather_cp": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen3_dense_profile_resolves_model_names():
    profile = get_true_on_policy_model_profile("Qwen3-4B")
    contract = get_true_on_policy_contract("true_on_policy_v1")

    assert profile.family == "qwen3_dense"
    assert profile.contract is TRUE_ON_POLICY_V1
    assert profile.contract is contract
    assert contract.name == "true_on_policy_v1"
    assert profile.supported_train_layouts == ("dp", "tp", "pp", "ulysses_cp")
    assert profile.supported_rollout_layouts == ("dp", "tp")
    assert profile.supports_ulysses_cp
    assert profile.supports_train_tensor_parallel
    assert profile.supports_rollout_tensor_parallel
    assert get_megatron_model_type("Qwen3-4B") == "qwen3-4B"
    assert get_megatron_model_type("Qwen3-4B-Instruct-2507") == "qwen3-4B-Instruct-2507"


def test_unknown_true_on_policy_model_fails_early():
    with pytest.raises(ValueError, match="does not have a model profile"):
        get_true_on_policy_model_profile("unknown-model")


@pytest.mark.parametrize(
    ("tp_size", "rollout_tp_size"),
    [
        (1, 1),
        (2, 1),
        (1, 2),
    ],
)
def test_launch_plan_does_not_vary_with_tensor_parallel_degree(
    tp_size: int,
    rollout_tp_size: int,
):
    """The program is declared, not derived from topology: a result at one degree transfers."""
    args = _args(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=rollout_tp_size,
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.sglang_rl_on_policy_target is None
    assert plan.contract is TRUE_ON_POLICY_V1
    assert plan.sglang_args.values == (
        "--sglang-enable-deterministic-inference",
        "--sglang-true-on-policy-contract",
        "true_on_policy_v1",
        "--sglang-attention-backend",
        "fa3",
    )
    assert "--sglang-rl-on-policy-target" not in plan.train_args


def test_legacy_sglang_target_override_is_ignored():
    args = _args(
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
        sglang_rl_on_policy_target="fsdp",
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.sglang_rl_on_policy_target == "fsdp"
    assert plan.kernel_policy is not None
    assert "--sglang-rl-on-policy-target" not in plan.train_args
    assert "ROW_LINEAR_ENABLE_INV" not in plan.env_vars


def test_contract_object_owns_miles_kernel_policy_values():
    args = _args(
        train_backend="megatron",
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
    )

    plan = build_true_on_policy_launch_plan(args)

    assert plan.kernel_policy is not None
    assert plan.kernel_policy.contract is TRUE_ON_POLICY_V1
    assert plan.kernel_policy.sglang_attention_backend == "fa3"
    assert plan.kernel_policy.build_megatron_args().values == (
        "--true-on-policy-contract",
        "true_on_policy_v1",
        "--spec",
        "miles_plugins.top.spec",
        "get_top_spec",
        "--transformer-impl",
        "local",
        "--use-cpu-initialization",
        "--batch-invariant-mode",
        "--no-bias-swiglu-fusion",
    )


def test_megatron_true_on_policy_disables_sequence_parallel_and_enables_backend_flags():
    args = _args(train_backend="megatron", use_sequence_parallel=True)

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is False
    assert "--use-sglang" not in plan.train_args
    assert "--true-on-policy-contract true_on_policy_v1" in plan.train_args
    assert "--sglang-true-on-policy-contract true_on_policy_v1" in plan.train_args
    # the gate scores DECODE-produced logprobs; a plan that re-adds the prefill recompute
    # certifies a different program (and masks every decode-only gap)
    assert "--recompute-logprobs-via-prefill" not in plan.train_args
    assert "--batch-invariant-mode" in plan.train_args
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
    assert args.sglang_rl_on_policy_target is None
    assert plan.parallel_layout is not None
    assert plan.parallel_layout.train_tensor_parallel_size == 2
    assert plan.parallel_layout.train_context_parallel_size == 4
    assert plan.parallel_layout.rollout_num_gpus_per_engine == 8
    assert plan.parallel_layout.uses_train_tp
    assert plan.parallel_layout.uses_ulysses_cp
    assert plan.parallel_layout.uses_rollout_tp
    assert plan.kernel_policy is not None
    assert plan.sglang_args.values == (
        "--sglang-enable-deterministic-inference",
        "--sglang-true-on-policy-contract",
        "true_on_policy_v1",
        "--sglang-attention-backend",
        "fa3",
    )
    assert plan.megatron_args.values == (
        "--true-on-policy-contract",
        "true_on_policy_v1",
        "--spec",
        "miles_plugins.top.spec",
        "get_top_spec",
        "--transformer-impl",
        "local",
        "--use-cpu-initialization",
        "--batch-invariant-mode",
        "--no-bias-swiglu-fusion",
    )
    assert plan.miles_args.values == (
        "--deterministic-mode",
        "--true-on-policy-mode",
        # emitted because this fixture is cp=4: megatron's parser defaults it to ["p2p"], so an
        # unemitted declaration would silently select ring
        "--cp-comm-type",
        "a2a",
    )
    assert plan.env_vars == {
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


def test_true_on_policy_contract_override_is_validated():
    args = _args(true_on_policy_contract="unknown_contract")

    with pytest.raises(ValueError, match="Unsupported true-on-policy contract"):
        build_true_on_policy_launch_plan(args)


def test_non_megatron_train_backend_is_refused():
    """Megatron is the true-on-policy backend; another backend must refuse, not lose its flags."""
    args = _args(
        train_backend="fsdp",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
    )

    with pytest.raises(ValueError, match="megatron backend only"):
        build_true_on_policy_launch_plan(args)


def test_fsdp_e2e_uses_current_true_on_policy_contract(monkeypatch):
    from tests.e2e.fsdp import test_qwen3_4B_fsdp_true_on_policy as fsdp_e2e

    captured = {}
    monkeypatch.setattr(fsdp_e2e.U, "get_default_wandb_args", lambda *args, **kwargs: "")
    monkeypatch.setattr(fsdp_e2e.U, "execute_train", lambda **kwargs: captured.update(kwargs))

    fsdp_e2e.execute()

    train_args = captured["train_args"]
    assert "--sglang-true-on-policy-contract true_on_policy_v1" in train_args
    assert "--recompute-logprobs-via-prefill" not in train_args
    assert "--sglang-rl-on-policy-target" not in train_args


def test_off_policy_builds_empty_launch_plan_and_does_not_mutate_args():
    args = _args(true_on_policy=False, use_sequence_parallel=True)

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is True
    assert args.sglang_rl_on_policy_target is None
    assert not plan.enabled
    assert plan.train_args == ""
    assert plan.env_vars == {}
