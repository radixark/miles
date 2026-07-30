from __future__ import annotations

import argparse
import dataclasses
import functools
import json
from argparse import Namespace
from typing import Any

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args as _args
from tests.fast.backends.sglang_utils.conftest import tiny_model_path

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.server_args_utils import (
    _UNCOMPARED_FIELDS,
    parse_server_args_argv,
    server_args_to_argv,
)
from miles.backends.sglang_utils.sglang_engine import _compute_server_args
from miles.utils.workers.argv_utils import _actions_by_dest, _render_action_argv, _resolve_action

_FIELDS_WITHOUT_A_RENDERABLE_CLI: dict[str, str] = {
    "custom_sigquit_handler": "A Python-only callable hook; sglang registers no CLI option for it.",
    "stat_loggers": "A Python-only injection point; sglang registers no CLI option for it.",
    "uses_mamba_radix_cache": "Derived inside __post_init__; sglang registers no CLI option for it.",
    "_speculative_draft_quantization_explicitly_set": (
        "Derived inside __post_init__ and declared Arg(no_cli=True); sglang registers no CLI option for it."
    ),
    "cuda_graph_config": (
        "The CLI parses a validated per-phase JSON object while ServerArgs holds a CudaGraphConfig "
        "instance, so no generically rendered token parses back to the same value."
    ),
}


def _server_args(
    *,
    worker_type: str = "regular",
    node_rank: int = 0,
    dist_init_addr: str = "10.0.0.1:20000",
    args: Namespace | None = None,
    sglang_overrides: dict | None = None,
    disaggregation_bootstrap_port: int | None = None,
    num_gpus_per_engine: int = 1,
) -> dict:
    # ServerArgs probes the local accelerator when no device is given, which a CPU-only
    # CI runner cannot answer. Production resolves it to the engine's own device the same way.
    overrides = {"device": "cuda", **(sglang_overrides or {})}
    server_args_dict = _compute_server_args(
        args or _args(),
        node_rank=node_rank,
        dist_init_addr=dist_init_addr,
        nccl_port=20031,
        host="10.0.0.1",
        port=30000,
        worker_type=worker_type,
        disaggregation_bootstrap_port=disaggregation_bootstrap_port,
        base_gpu_id=0,
        engine_info_bootstrap_port=20033,
        sglang_overrides=overrides,
        num_gpus_per_engine=num_gpus_per_engine,
    )
    return server_args_dict


def _assert_roundtrips(server_args_dict: dict) -> None:
    """Every field the launched process ends up with matches what miles asked for."""
    parsed = parse_server_args_argv(server_args_to_argv(server_args_dict))
    wanted = ServerArgs(**server_args_dict)
    differing = [
        field.name
        for field in dataclasses.fields(wanted)
        if field.name not in _UNCOMPARED_FIELDS and getattr(parsed, field.name) != getattr(wanted, field.name)
    ]
    assert differing == []


class TestServerArgsToArgv:
    def test_a_regular_engine_launch_roundtrips(self):
        """The exact ServerArgs the launch computes survives the argv boundary."""
        server_args = _server_args()
        assert "--disaggregation-mode" not in server_args_to_argv(server_args)
        _assert_roundtrips(server_args)

    def test_the_identity_flags_are_always_rendered_exactly_once(self):
        """Model path, addressing, and device must stay explicit even at CLI defaults."""
        argv = server_args_to_argv(_server_args())
        for flag in ("--trust-remote-code", "--model-path", "--host", "--port", "--device"):
            assert argv.count(flag) == 1

    def test_an_unspecified_device_renders_the_auto_detected_accelerator(self, monkeypatch):
        """An unset device renders the accelerator chosen by ServerArgs instead of the text None."""
        monkeypatch.setattr("sglang.srt.server_args.get_device", lambda: "cuda")
        server_args = _server_args(sglang_overrides={"device": None})
        argv = server_args_to_argv(server_args)

        assert server_args["device"] is None
        assert argv[argv.index("--device") + 1] == "cuda"
        _assert_roundtrips(server_args)

    def test_a_prefill_worker_roundtrips(self):
        """PD-disaggregation prefill fields survive the argv boundary."""
        server_args = _server_args(worker_type="prefill", disaggregation_bootstrap_port=20090)
        assert server_args["disaggregation_mode"] == "prefill"
        assert "--disaggregation-mode" in server_args_to_argv(server_args)
        _assert_roundtrips(server_args)

    def test_a_decode_worker_roundtrips(self):
        """PD-disaggregation decode fields survive the argv boundary."""
        server_args = _server_args(worker_type="decode")
        assert server_args["disaggregation_mode"] == "decode"
        _assert_roundtrips(server_args)

    def test_a_multi_node_rank_roundtrips(self):
        """nnodes, node_rank and tp_size of a multi-node engine survive the boundary."""
        server_args = _server_args(
            node_rank=1,
            num_gpus_per_engine=16,
            args=_args(rollout_num_gpus_per_engine=16),
        )
        assert server_args["nnodes"] == 2 and server_args["node_rank"] == 1
        _assert_roundtrips(server_args)

    def test_dtype_and_parallel_sizes_roundtrip(self):
        """fp16 and dp/pp/ep sizes land in the argv and parse back."""
        server_args = _server_args(args=_args(fp16=True, sglang_dp_size=2, sglang_ep_size=2))
        assert server_args["dtype"] == "float16"
        _assert_roundtrips(server_args)

    def test_dp_attention_defaults_are_normalized_exactly_once(self):
        """Raw DP inputs are compared before ServerArgs applies interacting defaults."""
        server_args = _server_args(
            args=_args(
                rollout_num_gpus_per_engine=2,
                sglang_dp_size=2,
                sglang_enable_dp_attention=True,
                sglang_chunked_prefill_size=4096,
                sglang_schedule_conservativeness=1.0,
            ),
            num_gpus_per_engine=2,
        )
        expected = ServerArgs(**server_args)
        argv = server_args_to_argv(server_args)
        parsed = parse_server_args_argv(argv)

        assert "--schedule-conservativeness" not in argv
        assert argv[argv.index("--chunked-prefill-size") + 1] == "4096"
        assert parsed.schedule_conservativeness == expected.schedule_conservativeness == 0.3
        assert parsed.chunked_prefill_size == expected.chunked_prefill_size == 2048
        _assert_roundtrips(server_args)

    def test_sglang_overrides_roundtrip(self):
        """User overrides merged into the dict survive the argv boundary."""
        server_args = _server_args(sglang_overrides={"mem_fraction_static": 0.5, "log_level": "warning"})
        assert server_args["mem_fraction_static"] == 0.5
        _assert_roundtrips(server_args)

    def test_lora_fields_roundtrip(self):
        """enable_lora, ranks and the target-modules list survive the boundary."""
        server_args = _server_args(args=_args(lora_rank=8, target_modules=["linear_qkv"]))
        assert server_args["enable_lora"]
        _assert_roundtrips(server_args)

    def test_an_ipv6_dist_init_addr_roundtrips(self):
        """The bracketed v6 rendezvous address survives the argv boundary."""
        server_args = _server_args(dist_init_addr="[fd00::1]:20000")
        _assert_roundtrips(server_args)

    def test_a_colocate_prefill_cuda_graph_backend_roundtrips(self):
        """Colocate forces the prefill cuda graph backend off, which sglang folds into a
        derived config object that has no faithful command-line spelling."""
        server_args = _server_args(sglang_overrides={"cuda_graph_backend_prefill": "disabled"})
        argv = server_args_to_argv(server_args)
        assert "--cuda-graph-config" not in argv
        _assert_roundtrips(server_args)

    def test_lora_adapter_paths_roundtrip(self):
        """The name=path lora mapping survives the argv boundary."""
        server_args = _server_args(
            args=_args(lora_rank=8, target_modules=["linear_qkv"], lora_adapter_path="/fake/adapter")
        )
        _assert_roundtrips(server_args)


class TestSglangPrefixedPassthrough:
    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            pytest.param("fp4_gemm_runner_backend", "marlin", id="cli-name-renamed-field"),
            pytest.param("preferred_sampling_params", {"temperature": 0.7}, id="json-parsed-field"),
            pytest.param("dllm_fdfo", False, id="boolean-optional-action-set-to-false"),
            pytest.param("disable_cuda_graph", True, id="field-reachable-only-by-a-legacy-alias"),
        ],
    )
    def test_a_prefixed_user_flag_reaches_the_engine_command(self, field_name: str, value: Any) -> None:
        """A --sglang-<field> value lands on the engine ServerArgs and survives the argv boundary."""
        server_args = _server_args(args=_args(**{f"sglang_{field_name}": value}))
        assert server_args[field_name] == value
        _assert_roundtrips(server_args)


class TestEveryServerArgsFieldIsRenderable:
    @pytest.mark.parametrize(
        "field_name",
        [
            (
                pytest.param(
                    field.name,
                    marks=pytest.mark.xfail(reason=_FIELDS_WITHOUT_A_RENDERABLE_CLI[field.name], strict=True),
                    id=field.name,
                )
                if field.name in _FIELDS_WITHOUT_A_RENDERABLE_CLI
                else pytest.param(field.name, id=field.name)
            )
            for field in dataclasses.fields(ServerArgs)
        ],
    )
    def test_a_field_renders_to_argv_that_parses_back_to_the_same_value(self, field_name: str) -> None:
        """Every ServerArgs field resolves to a CLI action that round-trips a value of its own shape."""
        parser = _make_server_args_parser()
        action = _resolve_action(_actions_by_dest(parser), field_name=field_name, field_to_dest={})
        default_value = getattr(_baseline_namespace(), action.dest, None)

        accepted = _first_roundtripping_value(action=action, default_value=default_value)

        assert (
            accepted is not None
        ), f"{field_name!r} renders to argv that {action.option_strings[0]!r} cannot parse back"


def _make_server_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser


@functools.lru_cache(maxsize=1)
def _baseline_namespace() -> Namespace:
    return _make_server_args_parser().parse_args(_model_argv())


def _model_argv() -> list[str]:
    return ["--model-path", str(tiny_model_path())]


def _first_roundtripping_value(*, action: argparse.Action, default_value: object) -> object | None:
    for value in _sweep_candidates(action=action, default_value=default_value):
        argv = _render_action_argv(action, value)
        try:
            namespace = _make_server_args_parser().parse_args([*_model_argv(), *argv])
        except SystemExit:
            continue
        if getattr(namespace, action.dest) == value:
            return value
    return None


def _sweep_candidates(*, action: argparse.Action, default_value: object) -> list[object]:
    if isinstance(action, argparse.BooleanOptionalAction):
        return [not bool(default_value)]

    if action.nargs == 0:
        return [action.const]

    if action.type is json.loads:
        return [{"sweep-key": "sweep-value"}]

    if getattr(action.type, "__name__", "") == "json_list_type":
        return [["sweep-value"]]

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        return [[element] for element in _scalar_candidates(action=action, default_value=None)]

    return _scalar_candidates(action=action, default_value=default_value)


def _scalar_candidates(*, action: argparse.Action, default_value: object) -> list[object]:
    if action.choices:
        return [next((choice for choice in action.choices if choice != default_value), default_value)]

    if action.type is int:
        return [default_value + 1 if isinstance(default_value, int) else 1]

    if action.type is float:
        return [default_value + 1.0 if isinstance(default_value, float) else 1.0]

    if action.type in (str, None):
        return ["other-sweep-value" if default_value == "sweep-value" else "sweep-value"]

    return [1, "sweep-value", 1.0]
