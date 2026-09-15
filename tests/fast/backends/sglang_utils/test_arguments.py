from __future__ import annotations

import argparse
import dataclasses

import pytest

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils import arguments as sglang_arguments
from miles.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides


def _sglang_flags() -> set[str]:
    parser = add_sglang_arguments(argparse.ArgumentParser())
    return {option for action in parser._actions for option in action.option_strings}


def _parse_sglang_args(argv: list[str]) -> argparse.Namespace:
    return add_sglang_arguments(argparse.ArgumentParser()).parse_args(argv)


class TestSglangModelRoutersDefault:
    def test_parsing_without_model_routers_sets_none(self):
        """Parsing without multi-policy routers exposes a safe None default."""
        args = _parse_sglang_args([])

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
        assert "gated_launch_port" in {field.name for field in dataclasses.fields(ServerArgs)}


def _hide_server_arg(monkeypatch, name: str) -> None:
    real_add_cli_args = ServerArgs.add_cli_args
    hidden_flag = "--" + name.replace("_", "-")

    def add_cli_args(parser):
        original_add_argument = parser.add_argument

        def add_argument(*name_or_flags, **kwargs):
            if hidden_flag in name_or_flags:
                return None
            return original_add_argument(*name_or_flags, **kwargs)

        parser.add_argument = add_argument
        try:
            return real_add_cli_args(parser)
        finally:
            parser.add_argument = original_add_argument

    monkeypatch.setattr(ServerArgs, "add_cli_args", add_cli_args)


class TestUnsupportedServerArgs:
    def test_prefill_weight_versions_defaults_to_false_when_sglang_lacks_the_field(self, monkeypatch):
        """An sglang without ServerArgs.enable_prefill_weight_versions still parses the flag as False."""
        _hide_server_arg(monkeypatch, "enable_prefill_weight_versions")

        args = _parse_sglang_args([])

        assert args.sglang_enable_prefill_weight_versions is False
        assert args.miles_owned_server_arg_fallbacks == ("enable_prefill_weight_versions",)

    def test_prefill_weight_versions_keeps_the_real_flag_when_sglang_has_the_field(self):
        """With a supporting sglang the real prefixed flag parses and no miles-owned fallback is registered."""
        if "enable_prefill_weight_versions" not in {field.name for field in dataclasses.fields(ServerArgs)}:
            pytest.skip("the installed sglang has no ServerArgs.enable_prefill_weight_versions")

        args = _parse_sglang_args(["--sglang-enable-prefill-weight-versions"])

        assert args.sglang_enable_prefill_weight_versions is True
        assert args.miles_owned_server_arg_fallbacks == ()

    def test_enabling_prefill_weight_versions_on_an_unsupported_sglang_is_an_error(self, monkeypatch):
        """The fallback flag only exists to keep parsing working, so requesting it must fail loudly."""
        _hide_server_arg(monkeypatch, "enable_prefill_weight_versions")
        args = _parse_sglang_args(["--sglang-enable-prefill-weight-versions"])

        with pytest.raises(ValueError, match="enable_prefill_weight_versions"):
            sglang_arguments._assert_supported_server_args_are_requested(args)
