from __future__ import annotations

import argparse
import dataclasses
from argparse import Namespace

import pytest
from sglang_router.launch_router import RouterArgs

from miles.backends.sglang_utils.router_args_utils import (
    compute_sglang_router_args,
    parse_router_args_argv,
    router_args_to_argv,
)

_ROUTER_FIELD_TO_DEST = {
    "server_cert_path": "tls_cert_path",
    "server_key_path": "tls_key_path",
    "prefill_urls": "prefill",
    "decode_urls": "decode",
}
_UNRENDERABLE_ROUTER_FIELDS = {
    "bootstrap_port_annotation": "The unprefixed parser does not register this dataclass-only field.",
    "control_plane_api_keys": "The parser normalizes API key tokens into tuples that cannot be serialized by the renderer.",
}
_ROUTER_FIELD_PARAMS: list[object] = [
    (
        pytest.param(
            field.name,
            marks=pytest.mark.xfail(reason=_UNRENDERABLE_ROUTER_FIELDS[field.name], strict=True),
            id=field.name,
        )
        if field.name in _UNRENDERABLE_ROUTER_FIELDS
        else pytest.param(field.name, id=field.name)
    )
    for field in dataclasses.fields(RouterArgs)
]


def _make_router_args(**overrides):
    router_args = parse_router_args_argv([])
    for name, value in overrides.items():
        setattr(router_args, name, value)
    return router_args


class TestRouterArgsToArgv:
    def test_defaults_render_to_an_empty_argv(self):
        """A default RouterArgs needs no flags at all."""
        assert router_args_to_argv(parse_router_args_argv([])) == []

    def test_typical_launch_fields_roundtrip(self):
        """The fields miles sets at launch survive the argv boundary."""
        router_args = _make_router_args(
            host="10.0.0.1",
            port=1234,
            prometheus_port=4001,
            log_level="warn",
            request_timeout_secs=600,
            pd_disaggregation=True,
        )
        argv = router_args_to_argv(router_args)
        assert "--pd-disaggregation" in argv
        assert parse_router_args_argv(argv) == router_args

    def test_list_and_dict_fields_roundtrip(self):
        """nargs list flags and key=value dict flags survive the boundary."""
        router_args = _make_router_args(
            worker_urls=["http://a:1", "http://b:2"],
            selector={"app": "sglang"},
        )
        assert parse_router_args_argv(router_args_to_argv(router_args)) == router_args

    def test_cli_only_defaults_are_not_rendered(self):
        """Fields keeping their CLI default (even when it differs from the
        dataclass default, e.g. prometheus_host) stay off the command line."""
        argv = router_args_to_argv(_make_router_args(port=1234))
        assert "--prometheus-host" not in argv

    @pytest.mark.parametrize("field_name", _ROUTER_FIELD_PARAMS)
    def test_each_router_field_roundtrips(self, field_name: str) -> None:
        """Every CLI-backed RouterArgs field survives rendering and parsing."""
        parser = _make_router_cli_parser()
        action = _find_router_action(parser=parser, field_name=field_name)
        router_args = parse_router_args_argv([])
        default_value = getattr(router_args, field_name)
        value = _make_non_default_router_value(
            field_name=field_name,
            action=action,
            default_value=default_value,
        )
        assert value != default_value

        setattr(router_args, field_name, value)

        assert parse_router_args_argv(router_args_to_argv(router_args)) == router_args


class TestComputeSglangRouterArgs:
    def _make_args(self, **overrides) -> Namespace:
        defaults = dict(sglang_router_request_timeout_secs=600, sglang_router_policy=None)
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_call_site_values_override_the_defaults(self):
        """Host, ports, log level, and timeout come from the call site and args."""
        router_args = compute_sglang_router_args(
            self._make_args(sglang_router_request_timeout_secs=123),
            host="10.0.0.1",
            port=1234,
            prometheus_port=4001,
            has_pd_disaggregation=False,
        )
        assert router_args.host == "10.0.0.1"
        assert router_args.port == 1234
        assert router_args.prometheus_port == 4001
        assert router_args.log_level == "warn"
        assert router_args.request_timeout_secs == 123
        assert router_args.pd_disaggregation is False

    def test_policy_and_pd_are_conditional(self):
        """The policy override and the PD flag only apply when requested."""
        router_args = compute_sglang_router_args(
            self._make_args(sglang_router_policy="power_of_two"),
            host="h",
            port=1,
            prometheus_port=2,
            has_pd_disaggregation=True,
        )
        assert router_args.policy == "power_of_two"
        assert router_args.pd_disaggregation is True

    def test_router_prefixed_user_flags_pass_through(self):
        """--router-* values on the miles args reach the RouterArgs."""
        router_args = compute_sglang_router_args(
            self._make_args(router_cache_threshold=0.75),
            host="h",
            port=1,
            prometheus_port=2,
            has_pd_disaggregation=False,
        )
        assert router_args.cache_threshold == 0.75


def _make_router_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return parser


def _find_router_action(*, parser: argparse.ArgumentParser, field_name: str) -> argparse.Action | None:
    dest = _ROUTER_FIELD_TO_DEST.get(field_name, field_name)
    return next((action for action in parser._actions if action.dest == dest), None)


def _make_non_default_router_value(
    *, field_name: str, action: argparse.Action | None, default_value: object
) -> object:
    if action is None:
        assert field_name in _UNRENDERABLE_ROUTER_FIELDS
        return "sweep-value"

    if isinstance(default_value, bool):
        return not default_value

    if isinstance(action, argparse._AppendAction):
        if action.nargs == "+":
            return [("http://prefill.invalid", 1)]
        assert action.nargs == 1
        return ["http://decode.invalid"]

    if field_name == "control_plane_api_keys":
        return [("sweep-id", "Sweep Name", "sweep-key", "admin")]

    if field_name == "jwt_role_mapping":
        return {"sweep-role": "admin"}

    if isinstance(default_value, dict):
        return {"sweep-key": "sweep-value"}

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        return [_make_non_default_scalar(action=action, default_value=None)]

    return _make_non_default_scalar(action=action, default_value=default_value)


def _make_non_default_scalar(*, action: argparse.Action, default_value: object) -> object:
    if action.choices:
        return next(choice for choice in action.choices if choice != default_value)

    if isinstance(default_value, bool):
        return not default_value

    if action.type is int:
        return default_value + 1 if isinstance(default_value, int) else 1

    if action.type is float:
        return default_value + 1.0 if isinstance(default_value, float) else 1.0

    return "other-sweep-value" if default_value == "sweep-value" else "sweep-value"
