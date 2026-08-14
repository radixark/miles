from __future__ import annotations

import argparse
from argparse import Namespace

import pytest
from sglang_router.launch_router import RouterArgs

from miles.backends.sglang_utils.router_args_utils import (
    compute_sglang_router_args,
    parse_router_args_argv,
    router_args_to_argv,
)


def _make_router_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return parser


def _make_prefixed_router_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    return parser


def _make_router_cli_values(**overrides: object) -> dict[str, object]:
    values = vars(_make_router_cli_parser().parse_args([]))
    values.update(overrides)
    return values


def _make_miles_args(**overrides: object) -> Namespace:
    values = vars(_make_prefixed_router_cli_parser().parse_args([]))
    values.update(sglang_router_request_timeout_secs=600, sglang_router_policy=None)
    values.update(overrides)
    return Namespace(**values)


class TestRouterArgsToArgv:
    def test_raw_cli_defaults_render_to_an_empty_argv(self) -> None:
        """Raw Router CLI defaults need no explicit flags."""
        assert router_args_to_argv(_make_router_cli_values()) == []

    def test_cli_defaults_with_different_resolved_shapes_stay_implicit(self) -> None:
        """Raw list and None defaults do not become spurious resolved-value flags."""
        values = _make_router_cli_values()
        resolved = RouterArgs.from_cli_args(Namespace(**values))

        assert values["jwt_role_mapping"] == []
        assert resolved.jwt_role_mapping == {}
        assert values["prometheus_host"] == "0.0.0.0"
        assert RouterArgs.__dataclass_fields__["prometheus_host"].default is None
        assert router_args_to_argv(values) == []

    def test_launch_values_roundtrip(self) -> None:
        """The fields Miles sets at launch survive the argv boundary."""
        values = _make_router_cli_values(
            host="10.0.0.1",
            port=1234,
            prometheus_port=4001,
            log_level="warn",
            request_timeout_secs=600,
            pd_disaggregation=True,
        )
        argv = router_args_to_argv(values)

        assert "--pd-disaggregation" in argv
        assert parse_router_args_argv(argv) == RouterArgs.from_cli_args(Namespace(**values))

    def test_cli_aliases_are_normalized_only_after_rendering(self) -> None:
        """TLS, prefill, and decode inputs keep parser names and raw shapes until parse."""
        values = _make_router_cli_values(
            tls_cert_path="/certs/server.pem",
            tls_key_path="/certs/server.key",
            prefill=[["http://prefill-a:8000", "9000"], ["http://prefill-b:8000", "none"]],
            decode=[["http://decode-a:8001"], ["http://decode-b:8001"]],
        )
        argv = router_args_to_argv(values)
        parsed = parse_router_args_argv(argv)

        assert argv.count("--prefill") == 2
        assert argv.count("--decode") == 2
        assert parsed.server_cert_path == "/certs/server.pem"
        assert parsed.server_key_path == "/certs/server.key"
        assert parsed.prefill_urls == [("http://prefill-a:8000", 9000), ("http://prefill-b:8000", None)]
        assert parsed.decode_urls == ["http://decode-a:8001", "http://decode-b:8001"]

    def test_special_collection_inputs_are_normalized_only_after_rendering(self) -> None:
        """Selectors and authentication collections cross the CLI in their raw token form."""
        values = _make_router_cli_values(
            selector=["app=sglang", "role=worker"],
            prefill_selector=["role=prefill"],
            decode_selector=["role=decode"],
            control_plane_api_keys=["key-id:Service Account:admin:secret"],
            jwt_role_mapping=["Gateway.Admin=admin", "Gateway.User=user"],
        )
        parsed = parse_router_args_argv(router_args_to_argv(values))

        assert parsed.selector == {"app": "sglang", "role": "worker"}
        assert parsed.prefill_selector == {"role": "prefill"}
        assert parsed.decode_selector == {"role": "decode"}
        assert parsed.control_plane_api_keys == [("key-id", "Service Account", "secret", "admin")]
        assert parsed.jwt_role_mapping == {"Gateway.Admin": "admin", "Gateway.User": "user"}

    @pytest.mark.parametrize(
        "dest",
        [
            pytest.param(action.dest, id=action.dest)
            for action in _make_router_cli_parser()._actions
            if action.option_strings and action.dest != "help"
        ],
    )
    def test_each_router_cli_action_roundtrips_from_its_raw_value(self, dest: str) -> None:
        """Every Router CLI action renders and parses from its native value domain."""
        parser = _make_router_cli_parser()
        action = next(action for action in parser._actions if action.dest == dest)
        values = vars(parser.parse_args([]))
        value = _make_non_default_raw_value(action=action)
        assert value != action.default
        values[dest] = value

        expected = RouterArgs.from_cli_args(Namespace(**values))

        assert parse_router_args_argv(router_args_to_argv(values)) == expected


class TestComputeSglangRouterArgs:
    def test_call_site_values_override_router_defaults(self) -> None:
        """Host, ports, log level, and timeout are applied in the raw value domain."""
        values = compute_sglang_router_args(
            _make_miles_args(sglang_router_request_timeout_secs=123),
            host="10.0.0.1",
            port=1234,
            prometheus_port=4001,
            has_pd_disaggregation=False,
        )

        assert values["host"] == "10.0.0.1"
        assert values["port"] == 1234
        assert values["prometheus_port"] == 4001
        assert values["log_level"] == "warn"
        assert values["request_timeout_secs"] == 123
        assert values["pd_disaggregation"] is False
        assert parse_router_args_argv(router_args_to_argv(values)).request_timeout_secs == 123

    def test_miles_policy_timeout_and_pd_overrides_take_precedence(self) -> None:
        """Miles-owned overrides win over their native --router-* counterparts."""
        values = compute_sglang_router_args(
            _make_miles_args(
                router_policy="random",
                router_request_timeout_secs=7,
                router_pd_disaggregation=False,
                sglang_router_policy="power_of_two",
                sglang_router_request_timeout_secs=123,
            ),
            host="h",
            port=1,
            prometheus_port=2,
            has_pd_disaggregation=True,
        )

        assert values["policy"] == "power_of_two"
        assert values["request_timeout_secs"] == 123
        assert values["pd_disaggregation"] is True

    def test_native_policy_and_pd_values_remain_without_miles_overrides(self) -> None:
        """Native Router values survive when Miles does not replace them."""
        values = compute_sglang_router_args(
            _make_miles_args(router_policy="bucket", router_pd_disaggregation=True),
            host="h",
            port=1,
            prometheus_port=2,
            has_pd_disaggregation=False,
        )

        assert values["policy"] == "bucket"
        assert values["pd_disaggregation"] is True

    def test_prefixed_inputs_are_extracted_without_resolving_aliases(self) -> None:
        """Prefixed Router inputs become unprefixed raw parser values, not dataclass values."""
        raw_prefill = [["http://prefill.invalid", "9000"]]
        raw_decode = [["http://decode.invalid"]]
        values = compute_sglang_router_args(
            _make_miles_args(
                router_cache_threshold=0.75,
                router_tls_cert_path="/certs/server.pem",
                router_prefill=raw_prefill,
                router_decode=raw_decode,
                router_jwt_role_mapping=["Gateway.Admin=admin"],
            ),
            host="h",
            port=1,
            prometheus_port=2,
            has_pd_disaggregation=False,
        )

        assert values["cache_threshold"] == 0.75
        assert values["tls_cert_path"] == "/certs/server.pem"
        assert values["prefill"] == raw_prefill
        assert values["decode"] == raw_decode
        assert values["jwt_role_mapping"] == ["Gateway.Admin=admin"]
        assert "server_cert_path" not in values
        assert "prefill_urls" not in values
        assert "decode_urls" not in values
        parsed = parse_router_args_argv(router_args_to_argv(values))
        assert parsed.server_cert_path == "/certs/server.pem"
        assert parsed.prefill_urls == [("http://prefill.invalid", 9000)]
        assert parsed.decode_urls == ["http://decode.invalid"]
        assert parsed.jwt_role_mapping == {"Gateway.Admin": "admin"}


def _make_non_default_raw_value(*, action: argparse.Action) -> object:
    if isinstance(action, argparse.BooleanOptionalAction):
        return not action.default

    if action.nargs == 0:
        assert action.const != action.default
        return action.const

    if action.dest == "prefill":
        return [["http://prefill.invalid", "9000"]]

    if action.dest == "decode":
        return [["http://decode.invalid"]]

    if action.dest == "control_plane_api_keys":
        return ["sweep-id:Sweep Name:admin:sweep-key"]

    if action.dest == "jwt_role_mapping":
        return ["sweep-role=admin"]

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        return [_make_non_default_raw_scalar(action=action)]

    return _make_non_default_raw_scalar(action=action)


def _make_non_default_raw_scalar(*, action: argparse.Action) -> object:
    if action.choices:
        return next(choice for choice in action.choices if choice != action.default)

    if action.type is int:
        return action.default + 1 if isinstance(action.default, int) else 1

    if action.type is float:
        return action.default + 1.0 if isinstance(action.default, float) else 1.0

    return "other-sweep-value" if action.default == "sweep-value" else "sweep-value"
