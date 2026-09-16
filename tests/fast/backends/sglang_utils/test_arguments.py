from __future__ import annotations

import argparse
import dataclasses

import pytest

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides


def _sglang_flags() -> set[str]:
    parser = add_sglang_arguments(argparse.ArgumentParser())
    return {option for action in parser._actions for option in action.option_strings}


def _parse_sglang_args(argv: list[str]) -> argparse.Namespace:
    return add_sglang_arguments(argparse.ArgumentParser()).parse_args(argv)


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
        assert "gated_launch_port" in {field.name for field in dataclasses.fields(ServerArgs)}
