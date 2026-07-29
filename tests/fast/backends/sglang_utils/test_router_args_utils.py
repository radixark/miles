from __future__ import annotations

from argparse import Namespace

from miles.backends.sglang_utils.router_args_utils import (
    compute_sglang_router_args,
    parse_router_args_argv,
    router_args_to_argv,
)


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
