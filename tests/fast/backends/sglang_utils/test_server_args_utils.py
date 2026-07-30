from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv, server_args_to_argv
from miles.backends.sglang_utils.sglang_engine import _compute_server_args
from miles.utils.workers.argv_utils import _actions_by_dest, _render_action_argv, _resolve_action

_TINY_MODEL_CONFIG: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 256,
    "max_position_embeddings": 2048,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 1000,
}

_FIELDS_WITHOUT_A_RENDERABLE_CLI: dict[str, str] = {
    "custom_sigquit_handler": "A Python-only callable hook; sglang registers no CLI option for it.",
    "stat_loggers": "A Python-only injection point; sglang registers no CLI option for it.",
    "uses_mamba_radix_cache": "Derived inside __post_init__; sglang registers no CLI option for it.",
    "cuda_graph_config": (
        "The CLI parses a validated per-phase JSON object while ServerArgs holds a CudaGraphConfig "
        "instance, so no generically rendered token parses back to the same value."
    ),
}


@functools.lru_cache(maxsize=1)
def _tiny_model_path() -> Path:
    model_path = Path(tempfile.mkdtemp(prefix="miles-tiny-model-"))
    (model_path / "config.json").write_text(json.dumps(_TINY_MODEL_CONFIG))
    return model_path


def _args(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = dict(
        hf_checkpoint=str(_tiny_model_path()),
        seed=42,
        offload_rollout=False,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_rank=0,
        lora_adapter_path=None,
        multi_lora=False,
        colocate=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _server_args(
    *,
    worker_type: str = "regular",
    rank: int = 0,
    dist_init_addr: str = "10.0.0.1:20000",
    args: Namespace | None = None,
    sglang_overrides: dict | None = None,
    disaggregation_bootstrap_port: int | None = None,
    num_gpus_per_engine: int = 1,
) -> ServerArgs:
    server_args_dict = _compute_server_args(
        args or _args(),
        rank,
        dist_init_addr,
        20031,
        "10.0.0.1",
        30000,
        worker_type=worker_type,
        disaggregation_bootstrap_port=disaggregation_bootstrap_port,
        base_gpu_id=0,
        engine_info_bootstrap_port=20033,
        sglang_overrides=sglang_overrides,
        num_gpus_per_engine=num_gpus_per_engine,
    )
    return ServerArgs(**server_args_dict)


def _roundtrip(server_args: ServerArgs) -> ServerArgs:
    return parse_server_args_argv(server_args_to_argv(server_args))


class TestServerArgsToArgv:
    def test_a_regular_engine_launch_roundtrips(self):
        """The exact ServerArgs the launch computes survives the argv boundary."""
        server_args = _server_args()
        assert _roundtrip(server_args) == server_args

    def test_the_identity_flags_are_always_rendered_exactly_once(self):
        """model path and addressing must be explicit on the command, even at CLI defaults."""
        argv = server_args_to_argv(_server_args())
        for flag in ("--model-path", "--host", "--port"):
            assert argv.count(flag) == 1

    def test_a_prefill_worker_roundtrips(self):
        """PD-disaggregation prefill fields survive the argv boundary."""
        server_args = _server_args(worker_type="prefill", disaggregation_bootstrap_port=20090)
        assert server_args.disaggregation_mode == "prefill"
        assert _roundtrip(server_args) == server_args

    def test_a_decode_worker_roundtrips(self):
        """PD-disaggregation decode fields survive the argv boundary."""
        server_args = _server_args(worker_type="decode")
        assert server_args.disaggregation_mode == "decode"
        assert _roundtrip(server_args) == server_args

    def test_a_multi_node_rank_roundtrips(self):
        """nnodes, node_rank and tp_size of a multi-node engine survive the boundary."""
        server_args = _server_args(
            rank=1,
            num_gpus_per_engine=16,
            args=_args(rollout_num_gpus_per_engine=16),
        )
        assert server_args.nnodes == 2 and server_args.node_rank == 1
        assert _roundtrip(server_args) == server_args

    def test_dtype_and_parallel_sizes_roundtrip(self):
        """fp16 and dp/pp/ep sizes land in the argv and parse back."""
        server_args = _server_args(args=_args(fp16=True, sglang_dp_size=2, sglang_ep_size=2))
        assert server_args.dtype == "float16"
        assert _roundtrip(server_args) == server_args

    def test_sglang_overrides_roundtrip(self):
        """User overrides merged into the dict survive the argv boundary."""
        server_args = _server_args(sglang_overrides={"mem_fraction_static": 0.5, "log_level": "warning"})
        assert server_args.mem_fraction_static == 0.5
        assert _roundtrip(server_args) == server_args

    def test_lora_fields_roundtrip(self):
        """enable_lora, ranks and the target-modules list survive the boundary."""
        server_args = _server_args(args=_args(lora_rank=8, target_modules=["linear_qkv"]))
        assert server_args.enable_lora
        assert _roundtrip(server_args) == server_args

    def test_an_ipv6_dist_init_addr_roundtrips(self):
        """The bracketed v6 rendezvous address survives the argv boundary."""
        server_args = _server_args(dist_init_addr="[fd00::1]:20000")
        assert _roundtrip(server_args) == server_args

    def test_lora_adapter_paths_roundtrip(self):
        """The name=path lora mapping survives the argv boundary."""
        server_args = _server_args(
            args=_args(lora_rank=8, target_modules=["linear_qkv"], lora_adapter_path="/fake/adapter")
        )
        assert server_args.lora_paths
        assert _roundtrip(server_args) == server_args


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
        assert getattr(server_args, field_name) == value
        assert _roundtrip(server_args) == server_args


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
        action = _resolve_action(_actions_by_dest(parser), field_name=field_name, dest_prefix="", field_to_dest={})
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
    return ["--model-path", str(_tiny_model_path())]


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
