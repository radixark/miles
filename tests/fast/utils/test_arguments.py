from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from miles.backends.sglang_utils.sglang_engine import _compute_server_args
from miles.utils.arguments import _maybe_apply_dumper_overrides, get_miles_extra_args_provider
from miles.utils.misc import function_registry

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]


def make_class_with_add_arguments():
    class MyFn:
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--my-custom-arg", type=int, default=42)

    return MyFn


def make_function_with_add_arguments():
    def my_fn():
        pass

    my_fn.add_arguments = lambda parser: parser.add_argument("--my-custom-arg", type=int, default=42)
    return my_fn


def make_function_without_add_arguments():
    def my_fn():
        pass

    return my_fn


@pytest.mark.parametrize("path_arg", PATH_ARGS)
class TestAddArgumentsSupport:

    @pytest.mark.parametrize("fn_factory", [make_class_with_add_arguments, make_function_with_add_arguments])
    def test_add_arguments_is_called_and_arg_is_parsed(self, path_arg, fn_factory):
        fn = fn_factory()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)


class TestMaybeApplyDumperOverrides:
    def _make_args(
        self,
        *,
        dumper_enable: bool = False,
        use_fault_tolerance: bool = False,
        router_disable_health_check: bool = False,
        rollout_health_check_interval: float = 30.0,
        start_rollout_id: int | None = None,
        num_rollout: int = 10,
        eval_interval: int | None = 5,
        save: str | None = "/tmp/checkpoint",
        save_interval: int | None = 5,
        save_retain_interval: int | None = 10,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dumper_enable=dumper_enable,
            use_fault_tolerance=use_fault_tolerance,
            router_disable_health_check=router_disable_health_check,
            rollout_health_check_interval=rollout_health_check_interval,
            start_rollout_id=start_rollout_id,
            num_rollout=num_rollout,
            eval_interval=eval_interval,
            save=save,
            save_interval=save_interval,
            save_retain_interval=save_retain_interval,
        )

    def test_noop_when_dumper_disabled(self) -> None:
        args = self._make_args(
            dumper_enable=False,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is True
        assert args.router_disable_health_check is False
        assert args.rollout_health_check_interval == 30.0
        assert args.num_rollout == 10
        assert args.eval_interval == 5
        assert args.save == "/tmp/checkpoint"
        assert args.save_interval == 5
        assert args.save_retain_interval == 10

    def test_disables_all_heartbeats(self) -> None:
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is False
        assert args.router_disable_health_check is True
        assert args.rollout_health_check_interval == 1e18

    def test_forces_single_rollout(self) -> None:
        args = self._make_args(dumper_enable=True, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.start_rollout_id == 0
        assert args.num_rollout == 1
        assert args.eval_interval is None
        assert args.save is None
        assert args.save_interval is None
        assert args.save_retain_interval is None

    def test_respects_start_rollout_id(self) -> None:
        args = self._make_args(dumper_enable=True, start_rollout_id=5, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 6


def test_recompute_logprobs_via_prefill_flag_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--recompute-logprobs-via-prefill"] + REQUIRED_ARGS)

    assert args.recompute_logprobs_via_prefill is True


@pytest.mark.parametrize(
    (
        "rollout_num_gpus_per_engine",
        "sglang_attention_context_parallel_size",
    ),
    [
        (1, 1),
        (8, 4),
    ],
)
def test_true_on_policy_args_propagate_to_sglang_server_args(
    rollout_num_gpus_per_engine: int,
    sglang_attention_context_parallel_size: int,
):
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=rollout_num_gpus_per_engine,
        sglang_data_parallel_size=1,
        sglang_pipeline_parallel_size=1,
        sglang_expert_parallel_size=1,
        sglang_attention_context_parallel_size=sglang_attention_context_parallel_size,
        sglang_enable_dp_attention=False,
        sglang_enable_dp_lm_head=False,
        sglang_router_policy=None,
        sglang_router_ip=None,
        true_on_policy_mode=True,
        recompute_logprobs_via_prefill=False,
        sglang_true_on_policy_contract="qwen3_dense_true_on_policy_v1",
        sglang_enable_deterministic_inference=False,
        sglang_enable_prefill_only_deterministic_inference=False,
        hf_checkpoint="hf://dummy",
        seed=7,
        offload_rollout=False,
        num_gpus_per_node=8,
        colocate=False,
        use_rollout_routing_replay=False,
        fp16=False,
    )

    sglang_validate_args(args)

    server_args, _ = _compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:12345",
        nccl_port=12346,
        host="127.0.0.1",
        port=30000,
        base_gpu_id=0,
    )

    assert args.sglang_enable_deterministic_inference is True
    assert args.sglang_enable_prefill_only_deterministic_inference is False
    assert args.sglang_enable_dp_lm_head is False
    assert "rl_on_policy_target" not in server_args
    assert server_args["true_on_policy_contract"] == "qwen3_dense_true_on_policy_v1"
    assert server_args["enable_deterministic_inference"] is True
    assert server_args.get("enable_prefill_only_deterministic_inference", False) is False
    assert server_args["enable_dp_lm_head"] is False


def test_true_on_policy_sglang_cp_dp_lm_head_overrides_engine_defaults():
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=8,
        sglang_data_parallel_size=1,
        sglang_pipeline_parallel_size=1,
        sglang_expert_parallel_size=1,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        sglang_attention_context_parallel_size=4,
        sglang_enable_dp_lm_head=True,
        true_on_policy_mode=True,
        hf_checkpoint="hf://dummy",
        seed=7,
        offload_rollout=False,
        num_gpus_per_node=8,
        colocate=False,
        use_rollout_routing_replay=False,
        fp16=False,
    )

    server_args, _ = _compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:12345",
        nccl_port=12346,
        host="127.0.0.1",
        port=30000,
        base_gpu_id=0,
    )

    assert server_args["enable_dp_lm_head"] is True
