import argparse
import dataclasses
import json
from typing import Any

import pytest
from pydantic import ValidationError

from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers import argv_utils
from miles.utils.workers.argv_utils import CONFIG_JSON_FLAG, config_to_argv, parse_config_argv, render_cli_argv


class _DemoConfig(FrozenStrictBaseModel):
    text: str
    count: int
    ratio: float
    enabled: bool
    maybe_timeout: float | None
    tags: list[str] | None
    options: dict[str, Any] | None


def _make_demo_config(**overrides) -> _DemoConfig:
    kwargs = dict(
        text="hello world",
        count=3,
        ratio=0.5,
        enabled=True,
        maybe_timeout=None,
        tags=["a", "b"],
        options={"nested": {"k": [1, 2]}},
    )
    kwargs.update(overrides)
    return _DemoConfig(**kwargs)


class TestConfigToArgv:
    def test_roundtrip_preserves_every_field_type(self):
        """str, int, float, bool, None, list, and nested dict all survive."""
        config = _make_demo_config()
        assert parse_config_argv(_DemoConfig, config_to_argv(config)) == config

    @pytest.mark.parametrize(
        "text",
        ["with space", 'quo"te', "single'quote", "中文字符", "line\nbreak", "--looks-like-a-flag", ""],
    )
    def test_roundtrip_survives_hostile_strings(self, text):
        """Quoting-hostile string values survive the argv boundary."""
        config = _make_demo_config(text=text)
        assert parse_config_argv(_DemoConfig, config_to_argv(config)).text == text

    def test_roundtrip_preserves_none_versus_value(self):
        """None and a real value on a nullable field stay distinguishable."""
        assert parse_config_argv(_DemoConfig, config_to_argv(_make_demo_config())).maybe_timeout is None
        config = _make_demo_config(maybe_timeout=30.0)
        assert parse_config_argv(_DemoConfig, config_to_argv(config)).maybe_timeout == 30.0

    def test_argv_is_a_flag_value_pair(self):
        """The rendered argv is exactly the config-json flag plus its payload."""
        argv = config_to_argv(_make_demo_config())
        assert argv[0] == CONFIG_JSON_FLAG
        assert len(argv) == 2

    def test_production_roundtrip_check_cannot_be_skipped(self, monkeypatch):
        """A parse that fails to reproduce the config aborts the render."""
        monkeypatch.setattr(argv_utils, "parse_config_argv", lambda config_cls, argv: _make_demo_config(count=999))
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            config_to_argv(_make_demo_config())

    def test_real_worker_configs_roundtrip(self):
        """The miles router and session server configs survive the boundary."""
        router_config = MilesRouterConfig(
            host="127.0.0.1",
            port=30080,
            max_connections=256,
            timeout=None,
            health_check_interval=10.0,
            health_check_failure_threshold=3,
        )
        assert parse_config_argv(MilesRouterConfig, config_to_argv(router_config)) == router_config

        session_config = SessionServerConfig(
            host="127.0.0.1",
            port=30100,
            instance_id="abc",
            backend_url="http://127.0.0.1:30000",
            timeout=600.0,
            hf_checkpoint="/fake/model",
            chat_template_path=None,
            tito_model="qwen3",
            apply_chat_template_kwargs={"enable_thinking": False},
            use_rollout_routing_replay=True,
            use_rollout_indexer_replay=False,
            sglang_speculative_algorithm=None,
            num_layers=None,
            moe_router_topk=None,
            save_debug_trajectory_data=None,
            lora_rank=0,
            lora_adapter_path=None,
            use_session_server="v2",
            session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_retries",
            session_sample_postprocessor_path=(
                "miles.rollout.session.v2.postprocessor_hub.default_postprocess"
            ),
        )
        assert parse_config_argv(SessionServerConfig, config_to_argv(session_config)) == session_config


class TestParseConfigArgv:
    def test_missing_flag_is_rejected(self):
        """An argv without the config-json flag fails to parse."""
        with pytest.raises(SystemExit):
            parse_config_argv(_DemoConfig, [])

    def test_unknown_flag_is_rejected(self):
        """Stray extra flags fail to parse instead of being ignored."""
        argv = config_to_argv(_make_demo_config())
        with pytest.raises(SystemExit):
            parse_config_argv(_DemoConfig, [*argv, "--unknown", "1"])

    def test_invalid_json_is_rejected(self):
        """A payload that is not valid JSON fails validation loudly."""
        with pytest.raises(ValidationError):
            parse_config_argv(_DemoConfig, [CONFIG_JSON_FLAG, "not json"])

    def test_extra_json_fields_are_rejected(self):
        """A payload with unknown fields violates the strict schema."""
        payload = _make_demo_config().model_dump_json().replace("{", '{"unknown_field": 1, ', 1)
        with pytest.raises(ValidationError):
            parse_config_argv(_DemoConfig, [CONFIG_JSON_FLAG, payload])


@dataclasses.dataclass
class _DemoArgs:
    name: str = "default-name"
    count: int = 0
    ratio: float = 1.0
    verbose: bool = False
    enabled: bool = True
    items: list[str] = dataclasses.field(default_factory=list)
    mapping: dict[str, str] = dataclasses.field(default_factory=dict)
    cli_filled: str | None = None


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="default-name")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enabled", action="store_true", default=True)
    parser.add_argument("--items", nargs="*", default=[])
    parser.add_argument("--mapping", nargs="*", default=[])
    parser.add_argument("--cli-filled", default="filled-by-cli")
    return parser


def _from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
    return _DemoArgs(
        name=parsed.name,
        count=parsed.count,
        ratio=parsed.ratio,
        verbose=parsed.verbose,
        enabled=parsed.enabled,
        items=list(parsed.items),
        mapping=dict(item.split("=", 1) for item in parsed.mapping),
        cli_filled=parsed.cli_filled,
    )


def _render(args_obj: _DemoArgs) -> list[str]:
    return render_cli_argv(args_obj, make_parser=_make_parser, from_parsed=_from_parsed)


def _parse(argv: list[str]) -> _DemoArgs:
    return _from_parsed(_make_parser().parse_args(argv))


def _make_cli_default_args(**overrides) -> _DemoArgs:
    args_obj = _parse([])
    for name, value in overrides.items():
        setattr(args_obj, name, value)
    return args_obj


class TestRenderCliArgv:
    def test_cli_defaults_render_to_an_empty_argv(self):
        """An object matching the CLI defaults needs no flags at all."""
        assert _render(_parse([])) == []

    def test_scalar_bool_list_and_dict_fields_roundtrip(self):
        """Every rendered field kind survives parse back to an equal object."""
        args_obj = _make_cli_default_args(
            name="other",
            count=3,
            ratio=0.5,
            verbose=True,
            items=["a", "b"],
            mapping={"k1": "v1", "k2": "v2"},
        )
        argv = _render(args_obj)
        assert "--verbose" in argv
        assert _parse(argv) == args_obj

    def test_cli_only_defaults_are_not_rendered(self):
        """A field keeping its CLI default (even when it differs from the
        dataclass default) stays off the command line."""
        argv = _render(_make_cli_default_args(count=3))
        assert "--cli-filled" not in argv

    def test_unrenderable_false_on_a_true_default_flag_fails_loudly(self):
        """A store-true flag whose CLI default is True cannot express False."""
        with pytest.raises(AssertionError, match="cannot be rendered"):
            _render(_make_cli_default_args(enabled=False))

    def test_roundtrip_mismatch_aborts_the_render(self):
        """A from_parsed that fails to reproduce the object aborts the render."""
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            render_cli_argv(
                _make_cli_default_args(count=3),
                make_parser=_make_parser,
                from_parsed=lambda parsed: _make_cli_default_args(count=999),
            )


@dataclasses.dataclass
class _AliasArgs:
    server_cert_path: str | None = None
    prefill_urls: list[tuple] = dataclasses.field(default_factory=list)
    dllm_fdfo: bool = True
    mm_process_config: dict[str, Any] | None = None
    plain: str = "default"


def _make_alias_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-tls-cert-path", default=None)
    parser.add_argument("--router-prefill", action="append", nargs="+", default=[])
    parser.add_argument("--router-dllm-fdfo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--router-mm-process-config", type=json.loads, default=None)
    parser.add_argument("--router-plain", default="default")
    return parser


def _alias_from_parsed(parsed: argparse.Namespace) -> _AliasArgs:
    return _AliasArgs(
        server_cert_path=parsed.router_tls_cert_path,
        prefill_urls=[(url, int(bootstrap_port)) for url, bootstrap_port in parsed.router_prefill],
        dllm_fdfo=parsed.router_dllm_fdfo,
        mm_process_config=parsed.router_mm_process_config,
        plain=parsed.router_plain,
    )


_ALIAS_FIELD_TO_DEST = {"server_cert_path": "router_tls_cert_path", "prefill_urls": "router_prefill"}


def _render_alias(args_obj: _AliasArgs) -> list[str]:
    return render_cli_argv(
        args_obj,
        make_parser=_make_alias_parser,
        from_parsed=_alias_from_parsed,
        dest_prefix="router_",
        field_to_dest=_ALIAS_FIELD_TO_DEST,
    )


class TestRenderCliArgvAgainstTheRealParserShape:
    """The renderer must take flag names and value shapes from the parser, not from field names."""

    def test_prefixed_dest_is_resolved_without_an_explicit_mapping(self):
        """A field whose dest is merely prefixed needs no entry in field_to_dest."""
        argv = _render_alias(_AliasArgs(plain="other"))
        assert argv == ["--router-plain", "other"]

    def test_aliased_field_renders_the_registered_flag(self):
        """A field name that differs from its flag renders the flag the parser actually accepts."""
        argv = _render_alias(_AliasArgs(server_cert_path="/certs/a.pem"))
        assert argv == ["--router-tls-cert-path", "/certs/a.pem"]

    def test_boolean_optional_action_can_express_false(self):
        """A BooleanOptionalAction defaulting to True renders its negative option."""
        argv = _render_alias(_AliasArgs(dllm_fdfo=False))
        assert argv == ["--no-router-dllm-fdfo"]

    def test_json_valued_option_renders_a_single_json_token(self):
        """A dict option parsed by json.loads renders one JSON document, not key=value pairs."""
        argv = _render_alias(_AliasArgs(mm_process_config={"image": {"max_pixels": 1}}))
        assert argv == ["--router-mm-process-config", '{"image": {"max_pixels": 1}}']

    def test_append_action_repeats_the_flag_per_entry(self):
        """An append option renders once per entry, spreading each entry's tokens."""
        argv = _render_alias(_AliasArgs(prefill_urls=[("http://a:1", 9000), ("http://b:2", 9001)]))
        assert argv == ["--router-prefill", "http://a:1", "9000", "--router-prefill", "http://b:2", "9001"]

    @pytest.mark.parametrize(
        "args_obj",
        [
            _AliasArgs(server_cert_path="/certs/a.pem"),
            _AliasArgs(dllm_fdfo=False),
            _AliasArgs(mm_process_config={"image": {"max_pixels": 1}}),
            _AliasArgs(prefill_urls=[("http://a:1", 9000)]),
        ],
        ids=["aliased", "boolean-optional-false", "json-dict", "append-list"],
    )
    def test_every_shape_survives_the_production_roundtrip(self, args_obj: _AliasArgs):
        """Each shape parses back to an equal object, which is what the production assert enforces."""
        assert _alias_from_parsed(_make_alias_parser().parse_args(_render_alias(args_obj))) == args_obj

    def test_a_field_with_no_registered_option_fails_loudly(self):
        """An unrenderable field is rejected instead of being rendered as a guessed flag."""

        @dataclasses.dataclass
        class _UnknownArgs:
            not_on_the_cli: str = "default"

        with pytest.raises(AssertionError, match="cannot be rendered"):
            render_cli_argv(
                _UnknownArgs(not_on_the_cli="other"),
                make_parser=_make_alias_parser,
                from_parsed=lambda parsed: _UnknownArgs(),
                dest_prefix="router_",
            )


@dataclasses.dataclass
class _RequiredDemoArgs:
    model: str
    count: int = 0


def _make_required_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=0)
    return parser


def _from_parsed_required(parsed: argparse.Namespace) -> _RequiredDemoArgs:
    return _RequiredDemoArgs(model=parsed.model, count=parsed.count)


class TestRenderCliArgvRequiredArgv:
    def test_required_flags_are_emitted_exactly_once(self):
        """required_argv seeds the defaults probe and stays in the final argv."""
        args_obj = _RequiredDemoArgs(model="m", count=3)
        argv = render_cli_argv(
            args_obj,
            make_parser=_make_required_parser,
            from_parsed=_from_parsed_required,
            required_argv=["--model", "m"],
        )
        assert argv.count("--model") == 1
        assert _from_parsed_required(_make_required_parser().parse_args(argv)) == args_obj

    def test_a_parser_with_required_flags_needs_required_argv(self):
        """Without required_argv the defaults probe hits the missing required flag."""
        with pytest.raises(SystemExit):
            render_cli_argv(
                _RequiredDemoArgs(model="m"),
                make_parser=_make_required_parser,
                from_parsed=_from_parsed_required,
            )
