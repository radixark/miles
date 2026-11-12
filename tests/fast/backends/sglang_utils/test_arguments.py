from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch


import pytest
from tests.fast.backends.sglang_utils.conftest import tiny_model_path

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

import miles.backends.sglang_utils.arguments as sglang_arguments
from miles.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides, validate_args
from miles.utils.arguments import parse_args
from miles.utils.workers.argv_utils import _record_field_names


def _sglang_flags() -> set[str]:
    parser = add_sglang_arguments(argparse.ArgumentParser())
    return {option for action in parser._actions for option in action.option_strings}


def _parse_sglang_args(argv: list[str]) -> argparse.Namespace:
    return add_sglang_arguments(argparse.ArgumentParser()).parse_args(argv)


class TestSglangModelRoutersDefault:
    def test_parsing_without_model_routers_sets_none(self):
        """Parsing without multi-policy routers exposes a safe None default."""
        argv = ["pytest", "--train-backend", "fsdp", "--rollout-batch-size", "1", "--num-rollout", "1"]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        assert args.sglang_model_routers is None


class TestAllocatorOwnedServerArgs:
    def test_the_launch_gate_port_is_not_exposed_on_the_cli(self):
        """The gate port comes from the addr allocator, so a flag for it could only point the engine elsewhere."""
        flags = _sglang_flags()

        assert "--sglang-gated-launch-port" not in flags
        assert "--eval-sglang-gated-launch-port" not in flags

    def test_a_tunable_server_arg_is_still_exposed(self):
        """The skip list must stay narrow: ordinary ServerArgs fields remain reachable from the cli."""
        assert "--sglang-mem-fraction-static" in _sglang_flags()

    def test_a_launch_gate_port_flag_is_a_hard_cli_error(self):
        """Passing a gate port on the command line must fail loudly instead of being accepted and ignored."""
        for flag in ("--sglang-gated-launch-port", "--eval-sglang-gated-launch-port"):
            with pytest.raises(SystemExit):
                _parse_sglang_args([flag, "13007"])

    def test_parsing_leaves_no_launch_gate_port_attribute_on_the_namespace(self):
        """The engine copies every args.sglang_<field> onto ServerArgs, so a parsed attribute would travel to it."""
        args = _parse_sglang_args([])

        assert not hasattr(args, "sglang_gated_launch_port")
        assert not hasattr(args, "eval_sglang_gated_launch_port")
        assert "gated_launch_port" not in collect_eval_sglang_overrides(args)

    def test_the_other_allocator_owned_server_args_stay_off_the_cli(self):
        """The gate port joins the existing skip list instead of replacing the endpoint entries already there."""
        flags = _sglang_flags()

        for flag in ("--sglang-port", "--sglang-nccl-port", "--sglang-base-gpu-id", "--eval-sglang-port"):
            assert flag not in flags

    def test_the_skipped_launch_gate_port_names_a_real_server_args_field(self):
        """A renamed upstream field would leave the skip entry stale and quietly re-expose the flag."""
        assert "gated_launch_port" in set(_record_field_names(ServerArgs))


@pytest.mark.parametrize("loss_type", ["score_centering", "policy_loss", "sft_loss", None])
@pytest.mark.parametrize("explicit_fp32", [False, True])
def test_score_centering_enables_fp32_lm_head(loss_type: str | None, explicit_fp32: bool) -> None:
    args = _parse_sglang_args(["--sglang-enable-fp32-lm-head"] if explicit_fp32 else [])
    args.loss_type = loss_type
    args.rollout_num_gpus_per_engine = 1
    args.true_on_policy_mode = False
    args.recompute_logprobs_via_prefill = False
    args.use_session_server = False
    args.sglang_router_policy = None
    args.sglang_router_ip = None

    validate_args(args)

    assert args.sglang_enable_fp32_lm_head is (explicit_fp32 or loss_type == "score_centering")


class TestParserMethodRestoration:
    def test_later_parser_extensions_use_class_method_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Registering SGLang flags must not hide later parser extensions behind a saved bound method."""
        parser = add_sglang_arguments(argparse.ArgumentParser())
        original = argparse.ArgumentParser.add_argument
        registered: list[str] = []

        def add_argument(self: argparse.ArgumentParser, name: str, **kwargs: object) -> argparse.Action:
            registered.append(name)
            return original(self, name, **kwargs)

        monkeypatch.setattr(argparse.ArgumentParser, "add_argument", add_argument)
        parser.add_argument("--later-extension", type=int)

        assert registered == ["--later-extension"]
        assert parser.parse_args(["--later-extension", "17"]).later_extension == 17


@pytest.mark.parametrize(
    ("platform_default", "flag", "expected"),
    [
        ("bfloat16", None, "bfloat16"),  # TPU: the engine's fp8 default is not used unless asked for
        ("bfloat16", "fp8_e4m3", "fp8_e4m3"),
        (None, None, "auto"),  # CUDA: the engine keeps its own choice
        (None, "fp8_e4m3", "fp8_e4m3"),
    ],
)
def test_rollout_kv_cache_dtype_defaults_to_the_platform_choice(
    monkeypatch: pytest.MonkeyPatch, platform_default: str | None, flag: str | None, expected: str
) -> None:
    """An explicit --sglang-kv-cache-dtype always wins; "auto" takes the platform's rollout default."""
    monkeypatch.setattr(sglang_arguments, "current_platform", SimpleNamespace(rollout_kv_cache_dtype=platform_default))
    args = _parse_sglang_args(["--sglang-kv-cache-dtype", flag] if flag else [])
    args.loss_type = None
    args.rollout_num_gpus_per_engine = 1
    args.true_on_policy_mode = False
    args.recompute_logprobs_via_prefill = False
    args.use_session_server = False
    args.sglang_router_policy = None
    args.sglang_router_ip = None

    validate_args(args)

    assert args.sglang_kv_cache_dtype == expected


class TestPipelinedEngineArgs:
    @staticmethod
    def _validated(argv: list[str], *, rollout_num_gpus_per_engine: int):
        args = _parse_sglang_args(argv)
        args.hf_checkpoint = str(tiny_model_path())  # 2 layers
        args.loss_type = None
        args.rollout_num_gpus_per_engine = rollout_num_gpus_per_engine
        args.true_on_policy_mode = False
        args.recompute_logprobs_via_prefill = False
        args.use_session_server = False
        args.sglang_router_policy = None
        args.sglang_router_ip = None
        validate_args(args)
        return args

    def test_the_recorded_sglang_tp_size_is_per_pipeline_stage(self):
        args = self._validated(["--sglang-pp-size", "8"], rollout_num_gpus_per_engine=64)

        assert args.sglang_tp_size == 8

    def test_a_partition_with_one_entry_per_stage_summing_to_the_layer_count_is_accepted(self):
        args = self._validated(
            ["--sglang-pp-size", "2", "--sglang-pp-layer-partition", "1,1"], rollout_num_gpus_per_engine=16
        )

        assert args.sglang_pp_layer_partition == "1,1"

    def test_a_partition_with_the_wrong_number_of_stages_is_refused(self):
        with pytest.raises(AssertionError, match="3 stages.*--sglang-pp-size 2"):
            self._validated(
                ["--sglang-pp-size", "2", "--sglang-pp-layer-partition", "1,1,0"], rollout_num_gpus_per_engine=16
            )

    def test_a_partition_that_does_not_cover_every_layer_is_refused(self):
        with pytest.raises(AssertionError, match="3 layers.*has 2"):
            self._validated(
                ["--sglang-pp-size", "2", "--sglang-pp-layer-partition", "2,1"], rollout_num_gpus_per_engine=16
            )

    def test_a_partition_that_is_not_a_list_of_layer_counts_is_refused(self):
        with pytest.raises(ValueError, match="comma-separated layer counts"):
            self._validated(
                ["--sglang-pp-size", "2", "--sglang-pp-layer-partition", "1;1"], rollout_num_gpus_per_engine=16
            )
