from types import SimpleNamespace

import pytest

from train_tinker import (
    _apply_api_key_environment,
    _configure_megatron_batch_placeholder,
    _validate_args,
)


def _args(**overrides):
    values = {
        "train_backend": "megatron",
        "multi_lora_n_adapters": 2,
        "indep_dp": False,
        "colocate": False,
        "use_fault_tolerance": False,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "qkv_format": "thd",
        "calculate_per_token_loss": False,
        "lora_dropout": 0.0,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "tinker_max_concurrent_samples": 256,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "tensor_model_parallel_size": 1,
        "micro_batch_size": 1,
        "global_batch_size": 1,
        "num_rollout": 1,
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_valid_service_arguments_are_accepted():
    _validate_args(_args())


def test_api_key_environment_is_applied_as_a_redacted_secret(monkeypatch):
    args = _args(tinker_api_key=None)
    monkeypatch.setenv("TINKER_API_KEY", "environment-secret")

    _apply_api_key_environment(args)

    assert args.tinker_api_key == "environment-secret"
    assert str(args.tinker_api_key) == "<redacted>"


def test_megatron_placeholder_batch_is_divisible_by_data_parallel_size():
    args = _args(
        actor_num_gpus_per_node=4,
        tensor_model_parallel_size=2,
        global_batch_size=1,
    )

    _configure_megatron_batch_placeholder(args)

    assert args.global_batch_size == 2
    assert args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_backend": "fsdp"}, "--train-backend must be megatron"),
        ({"indep_dp": True}, "--indep-dp is not supported"),
        ({"colocate": True}, "must be disaggregated"),
        ({"qkv_format": "bshd"}, "--qkv-format must be thd"),
    ],
)
def test_unsupported_service_topologies_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _validate_args(_args(**overrides))
