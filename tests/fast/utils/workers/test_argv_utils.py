import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers import argv_utils
from miles.utils.workers.argv_utils import (
    CONFIG_JSON_FLAG,
    config_to_argv,
    dataclass_to_values,
    parse_config_argv,
    render_cli_argv,
)


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
            session_message_matcher="strict",
            pause_generation_mode=None,
            session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_retries",
            session_sample_postprocessor_path=("miles.rollout.session.v2.postprocessor_hub.default_postprocess"),
        )
        assert parse_config_argv(SessionServerConfig, config_to_argv(session_config)) == session_config


class TestParseConfigArgv:
    def test_none_argv_parses_the_process_arguments(self, monkeypatch):
        """A None argv reads the payload from the process command line."""
        config = _make_demo_config()
        monkeypatch.setattr(sys, "argv", ["prog", *config_to_argv(config)])
        assert parse_config_argv(_DemoConfig, None) == config

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
    return render_cli_argv(
        _input_values(args_obj), expected_obj=args_obj, make_parser=_make_parser, from_parsed=_from_parsed
    )


def _parse(argv: list[str]) -> _DemoArgs:
    return _from_parsed(_make_parser().parse_args(argv))


def _make_cli_default_args(**overrides) -> _DemoArgs:
    args_obj = _parse([])
    for name, value in overrides.items():
        setattr(args_obj, name, value)
    return args_obj


def _input_values(args_obj: _DemoArgs) -> dict[str, object]:
    values = dataclass_to_values(args_obj)
    values["mapping"] = [f"{key}={value}" for key, value in args_obj.mapping.items()]
    return values


def _from_parsed_drifting(parsed: argparse.Namespace) -> _DemoArgs:
    return dataclasses.replace(_from_parsed(parsed), ratio=99.0)


def _render_drifting(args_obj: _DemoArgs, **overrides) -> list[str]:
    return render_cli_argv(
        _input_values(args_obj),
        expected_obj=args_obj,
        make_parser=_make_parser,
        from_parsed=_from_parsed_drifting,
        **overrides,
    )


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

    def test_a_variadic_dict_renders_key_value_tokens(self):
        """A dict handed to an nargs option becomes key=value tokens, not JSON."""
        args_obj = _make_cli_default_args(mapping={"k1": "v1", "k2": "v2"})
        argv = render_cli_argv(
            {"mapping": {"k1": "v1", "k2": "v2"}},
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=_from_parsed,
        )
        assert argv == ["--mapping", "k1=v1", "k2=v2"]

    def test_cli_only_defaults_are_not_rendered(self):
        """A field keeping its CLI default (even when it differs from the
        dataclass default) stays off the command line."""
        argv = _render(_make_cli_default_args(count=3))
        assert "--cli-filled" not in argv

    def test_none_constructor_inputs_are_left_for_the_cli_to_normalize(self):
        """A nullable input can normalize to a collection without being rendered."""
        args_obj = _parse([])
        argv = render_cli_argv(
            {"items": None},
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=_from_parsed,
        )
        assert argv == []

    def test_constructor_values_are_rendered_before_post_parse_normalization(self):
        """Raw values are not normalized twice when from_parsed rewrites them."""

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            args_obj = _from_parsed(parsed)
            if args_obj.verbose:
                args_obj.count //= 2
            return args_obj

        input_values = {**_input_values(_make_cli_default_args(verbose=True)), "count": 6}
        args_obj = from_parsed(_make_parser().parse_args(["--verbose", "--count", "6"]))
        argv = render_cli_argv(
            input_values,
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert _make_parser().parse_args(argv).count == 6
        assert from_parsed(_make_parser().parse_args(argv)) == args_obj

    def test_each_non_default_input_is_rendered_in_one_pass(self):
        """Interacting non-default inputs are both emitted without reconciliation."""

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            args_obj = _from_parsed(parsed)
            if args_obj.ratio == 1.0:
                args_obj.count = 3
            return args_obj

        input_values = {**_input_values(_make_cli_default_args(count=3)), "ratio": 0.5}
        args_obj = from_parsed(_make_parser().parse_args(["--count", "3", "--ratio", "0.5"]))
        argv = render_cli_argv(
            input_values,
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert argv == ["--count", "3", "--ratio", "0.5"]
        assert from_parsed(_make_parser().parse_args(argv)) == args_obj

    def test_raw_parser_default_is_omitted_before_post_parse_normalization(self):
        """A raw default is omitted so post-parse normalization runs exactly once."""

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            args_obj = _from_parsed(parsed)
            if args_obj.verbose:
                args_obj.ratio *= 0.3
            return args_obj

        input_values = {**_input_values(_make_cli_default_args(verbose=True)), "ratio": 1.0}
        expected_obj = from_parsed(_make_parser().parse_args(["--verbose"]))
        argv = render_cli_argv(
            input_values,
            expected_obj=expected_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert argv == ["--verbose"]
        assert from_parsed(_make_parser().parse_args(argv)) == expected_obj

    def test_expected_object_is_constructed_only_once_during_render(self):
        """The renderer performs one final conversion and never reconciles iteratively."""
        conversion_count = 0

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            nonlocal conversion_count
            conversion_count += 1
            return _from_parsed(parsed)

        expected_obj = _from_parsed(_make_parser().parse_args(["--count", "3"]))
        argv = render_cli_argv(
            {**_input_values(expected_obj), "count": 3},
            expected_obj=expected_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert argv == ["--count", "3"]
        assert conversion_count == 1

    def test_default_store_true_value_is_expressed_by_omitting_the_flag(self):
        """A default False remains implicit when other inputs change its resolved baseline."""

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            args_obj = _from_parsed(parsed)
            args_obj.verbose = args_obj.count == 0
            return args_obj

        input_values = _input_values(_make_cli_default_args(count=1))
        args_obj = from_parsed(_make_parser().parse_args(["--count", "1"]))
        argv = render_cli_argv(
            input_values,
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert "--verbose" not in argv
        assert from_parsed(_make_parser().parse_args(argv)) == args_obj

    def test_unrenderable_false_on_a_true_default_flag_fails_loudly(self):
        """A store-true flag whose CLI default is True cannot express False."""
        with pytest.raises(AssertionError, match="cannot be rendered"):
            _render(_make_cli_default_args(enabled=False))

    def test_roundtrip_mismatch_aborts_the_render(self):
        """A from_parsed that fails to reproduce the object aborts the render."""
        args_obj = _make_cli_default_args(count=3)
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            render_cli_argv(
                _input_values(args_obj),
                expected_obj=args_obj,
                make_parser=_make_parser,
                from_parsed=lambda parsed: _make_cli_default_args(count=999),
            )

    def test_a_field_the_parser_rewrites_blocks_the_render(self):
        """This is the failure that uncompared_fields exists to excuse."""
        args_obj = _make_cli_default_args(count=3)
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            _render_drifting(args_obj)

    def test_an_uncompared_field_is_excused_from_the_roundtrip(self):
        """Some upstream fields are rewritten on every parse and can never be made to match."""
        args_obj = _make_cli_default_args(count=3)
        argv = _render_drifting(args_obj, uncompared_fields=frozenset({"ratio"}))
        assert argv == ["--count", "3"]


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


_ALIAS_FIELD_TO_DEST = {
    "server_cert_path": "router_tls_cert_path",
    "prefill_urls": "router_prefill",
    "dllm_fdfo": "router_dllm_fdfo",
    "mm_process_config": "router_mm_process_config",
    "plain": "router_plain",
}


def _render_alias(args_obj: _AliasArgs) -> list[str]:
    return render_cli_argv(
        dataclass_to_values(args_obj),
        expected_obj=args_obj,
        make_parser=_make_alias_parser,
        from_parsed=_alias_from_parsed,
        field_to_dest=_ALIAS_FIELD_TO_DEST,
    )


class TestRenderCliArgvAgainstTheRealParserShape:
    """The renderer must take flag names and value shapes from the parser, not from field names."""

    def test_a_mapped_field_renders_the_dest_it_points_at(self):
        """The mapping is the only thing that connects a field name to a flag."""
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

        args_obj = _UnknownArgs(not_on_the_cli="other")
        with pytest.raises(AssertionError, match="cannot be rendered"):
            render_cli_argv(
                dataclass_to_values(args_obj),
                expected_obj=args_obj,
                make_parser=_make_alias_parser,
                from_parsed=lambda parsed: _UnknownArgs(),
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


def _render_required(args_obj: _RequiredDemoArgs, **overrides) -> list[str]:
    return render_cli_argv(
        dataclass_to_values(args_obj),
        expected_obj=args_obj,
        make_parser=_make_required_parser,
        from_parsed=_from_parsed_required,
        **overrides,
    )


class TestRenderCliArgvAlwaysRenderFields:
    def test_an_always_render_field_is_emitted_exactly_once(self):
        """An explicit policy field appears once even when another value is rendered."""
        args_obj = _RequiredDemoArgs(model="m", count=3)
        argv = _render_required(args_obj, always_render_fields=("model",))
        assert argv.count("--model") == 1
        assert _from_parsed_required(_make_required_parser().parse_args(argv)) == args_obj

    def test_an_always_render_field_is_emitted_even_at_its_own_default(self):
        """The explicit-output policy is independent of the parser default."""
        assert _render_required(_RequiredDemoArgs(model="m"), always_render_fields=("model",)) == ["--model", "m"]

    def test_an_unspecified_always_render_field_uses_the_resolved_value(self):
        """A raw None falls back to the value resolved by the target constructor."""
        argv = render_cli_argv(
            {"model": None},
            expected_obj=_RequiredDemoArgs(model="m"),
            make_parser=_make_required_parser,
            from_parsed=_from_parsed_required,
            always_render_fields=("model",),
        )
        assert argv == ["--model", "m"]

    def test_an_always_render_field_is_emitted_at_the_parser_default(self):
        """A value equal to the parser default is still emitted when the field is always rendered."""
        args_obj = _RequiredDemoArgs(model="m", count=0)
        assert _render_required(args_obj, always_render_fields=("count",)) == ["--count", "0", "--model", "m"]

    def test_an_always_render_field_missing_from_inputs_uses_the_expected_value(self):
        """An always-rendered field absent from the inputs falls back to the expected object."""
        args_obj = _RequiredDemoArgs(model="m", count=3)
        argv = render_cli_argv(
            {"count": 3},
            expected_obj=args_obj,
            make_parser=_make_required_parser,
            from_parsed=_from_parsed_required,
            always_render_fields=("model",),
        )
        assert argv == ["--model", "m", "--count", "3"]

    def test_an_always_render_field_prefers_the_raw_input_value(self):
        """A normalized expected value cannot replace its raw constructor input."""

        def from_parsed(parsed: argparse.Namespace) -> _RequiredDemoArgs:
            return _RequiredDemoArgs(model=parsed.model, count=parsed.count // 2)

        argv = render_cli_argv(
            {"model": "m", "count": 6},
            expected_obj=_RequiredDemoArgs(model="m", count=3),
            make_parser=_make_required_parser,
            from_parsed=from_parsed,
            always_render_fields=("count",),
        )
        assert argv == ["--count", "6", "--model", "m"]


_REPO_ROOT = Path(__file__).parents[4]


class TestPythonArgvPrefix:
    def _run_prefix_under(self, interpreter_flags: list[str]) -> list[str]:
        script = "import json, sys; from miles.utils.workers.argv_utils import python_argv_prefix; print(json.dumps(python_argv_prefix()))"
        completed = subprocess.run(
            [sys.executable, *interpreter_flags, "-c", script],
            capture_output=True,
            check=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        )
        return json.loads(completed.stdout)

    def test_a_plain_interpreter_yields_only_the_executable(self):
        """Nothing to preserve means the prefix is exactly what the old hardcoded rebuild produced."""
        assert self._run_prefix_under([]) == [sys.executable]

    def test_optimization_and_unbuffered_flags_are_preserved(self):
        """A re-executed child that drops -O runs with assertions back on, silently changing its semantics."""
        assert self._run_prefix_under(["-O", "-u"]) == [sys.executable, "-O", "-u"]

    def test_a_flag_taking_a_separate_value_keeps_its_value(self):
        """-X and its value are one option, so splitting them would feed the value to the module as an argument."""
        assert self._run_prefix_under(["-X", "faulthandler"]) == [sys.executable, "-X", "faulthandler"]

    def test_the_double_dash_terminator_is_not_forwarded(self, tmp_path: Path):
        """Forwarding -- would swallow the -m the caller appends, turning the module name into a script path."""
        script = tmp_path / "print_prefix.py"
        script.write_text(
            "import json\n"
            "from miles.utils.workers.argv_utils import python_argv_prefix\n"
            "print(json.dumps(python_argv_prefix()))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "--", str(script)],
            capture_output=True,
            check=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        )

        assert json.loads(completed.stdout) == [sys.executable, "-O"]

    def test_the_script_selector_and_everything_after_it_is_dropped(self):
        """The prefix is only the interpreter part; the caller supplies its own -m and module arguments."""
        prefix = self._run_prefix_under(["-O"])

        assert "-c" not in prefix and "-m" not in prefix

    def test_a_warning_filter_flag_keeps_its_value(self):
        """-W and its filter are one option, so dropping the filter both loses it and feeds it to the module."""
        assert _run_prefix_printing_command([sys.executable, "-W", "ignore", "-c", _PRINT_PREFIX_SOURCE]) == [
            sys.executable,
            "-W",
            "ignore",
        ]

    def test_a_hash_based_pyc_flag_keeps_its_long_form_value(self):
        """A long option taking a separate value is the case a short-flag-only scan silently mangles."""
        assert _run_prefix_printing_command(
            [sys.executable, "--check-hash-based-pycs", "always", "-c", _PRINT_PREFIX_SOURCE]
        ) == [sys.executable, "--check-hash-based-pycs", "always"]

    def test_flags_after_a_valued_flag_are_still_collected_in_order(self):
        """Consuming a flag's value must not stop the scan, or every later flag is lost from the child."""
        assert self._run_prefix_under(["-O", "-X", "faulthandler", "-u"]) == [
            sys.executable,
            "-O",
            "-X",
            "faulthandler",
            "-u",
        ]

    def test_a_module_run_stops_before_the_module_selector(self, tmp_path: Path):
        """Forwarding the parent's -m and module name would launch the parent's module instead of the child's."""
        module_path = tmp_path / "print_argv_prefix_module.py"
        module_path.write_text(_PRINT_PREFIX_SOURCE)

        prefix = _run_prefix_printing_command(
            [sys.executable, "-O", "-m", "print_argv_prefix_module"],
            extra_python_path=tmp_path,
        )

        assert prefix == [sys.executable, "-O"]

    def test_a_script_path_ends_the_prefix_and_its_arguments_are_not_absorbed(self, tmp_path: Path):
        """Script arguments that look like flags must not be mistaken for interpreter flags of the child."""
        script_path = tmp_path / "print_argv_prefix_script.py"
        script_path.write_text(_PRINT_PREFIX_SOURCE)

        prefix = _run_prefix_printing_command([sys.executable, "-O", str(script_path), "-u", "--verbose"])

        assert prefix == [sys.executable, "-O"]

    def test_reading_the_program_from_stdin_ends_the_prefix(self):
        """The stdin selector is not an interpreter flag, so forwarding it would make the child read stdin too."""
        prefix = _run_prefix_printing_command([sys.executable, "-O", "-"], stdin_text=_PRINT_PREFIX_SOURCE)

        assert prefix == [sys.executable, "-O"]


_PRINT_PREFIX_SOURCE = (
    "import json\n"
    "from miles.utils.workers.argv_utils import python_argv_prefix\n"
    "print(json.dumps(python_argv_prefix()))\n"
)


def _run_prefix_printing_command(
    command: list[str],
    *,
    extra_python_path: Path | None = None,
    stdin_text: str | None = None,
) -> list[str]:
    python_path_entries = [str(_REPO_ROOT)] + ([str(extra_python_path)] if extra_python_path is not None else [])
    completed = subprocess.run(
        command,
        capture_output=True,
        check=True,
        text=True,
        input=stdin_text,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(python_path_entries)},
    )
    return json.loads(completed.stdout)
