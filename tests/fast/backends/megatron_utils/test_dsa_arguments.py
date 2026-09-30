import argparse
import ast
import importlib.util
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.debug_utils.run_megatron.worker.script_args import WORKER_SCRIPT_ARGS_BRIDGE
from miles_plugins.models.deepseek_v4.arguments import add_dsv4_arguments
from miles_plugins.models.glm5.arguments import (
    MEGATRON_DSA_SPEC,
    MILES_DSA_SPEC,
    add_dsa_arguments,
    normalize_dsa_args,
)


def _args(**overrides):
    values = dict(
        dsa_impl="megatron",
        spec=list(MILES_DSA_SPEC),
        megatron_to_hf_mode="raw",
        context_parallel_size=1,
        allgather_cp=False,
        dsa_kernel_backend="cudnn",
        dsa_indexer_loss_coeff=None,
        miles_dsa_topk_backend="torch",
        cp_comm_type=None,
    )
    return Namespace(**(values | overrides))


def _hf_config(**overrides):
    values = dict(model_type="deepseek_v32", index_n_heads=64, index_head_dim=128, index_topk=2048)
    return SimpleNamespace(**(values | overrides))


def test_default_preserves_the_existing_miles_path():
    args = add_dsa_arguments(ArgumentParser()).parse_args([])
    assert vars(args) == {"dsa_impl": "miles", "miles_dsa_topk_backend": "torch", "cp_comm_type": None}
    before = vars(args).copy()
    normalize_dsa_args(args, None)
    assert vars(args) == before


@pytest.mark.parametrize(
    ("hf_overrides", "interleaved", "frequency", "offset"),
    [
        ({"model_type": "deepseek_v32"}, False, 1, 0),
        ({"model_type": "glm_moe_dsa", "indexer_rope_interleave": True}, True, 1, 0),
        (
            {
                "model_type": "glm_moe_dsa",
                "indexer_rope_interleave": True,
                "index_topk_freq": 4,
                "index_skip_topk_offset": 2,
            },
            True,
            4,
            2,
        ),
    ],
    ids=["deepseek-v32", "glm5", "glm52-shared-indices"],
)
def test_native_dsa_preserves_checkpoint_indexer_conventions(hf_overrides, interleaved, frequency, offset):
    args = _args()
    hf_config = _hf_config(**hf_overrides)
    normalize_dsa_args(args, hf_config)

    assert args.spec == list(MEGATRON_DSA_SPEC)
    assert args.experimental_attention_variant == "dsa"
    assert args.enable_experimental is True
    assert (args.dsa_indexer_n_heads, args.dsa_indexer_head_dim, args.dsa_indexer_topk) == (64, 128, 2048)
    assert args.dsa_indexer_rope_interleaved is interleaved
    assert args.indexer_rope_interleave is interleaved
    assert (args.dsa_indexer_topk_freq, args.dsa_indexer_skip_topk_offset) == (frequency, offset)
    assert args.dsa_indexer_rotate_activation is False
    assert args.dsa_indexer_k_norm_epsilon == 1e-6
    assert args.dsa_indexer_k_norm_fp32 is True
    assert args.dsa_kernel_backend == "cudnn"
    assert args.dsa_indexer_loss_coeff == 0.0
    assert args.dsa_indexer_topk_backend == "torch"

    # Checkpoint conversion and training can both normalize an already native spec.
    before = vars(args).copy()
    normalize_dsa_args(args, hf_config)
    assert vars(args) == before


@pytest.mark.parametrize("loss_coeff", [0.0, 0.001])
def test_explicit_indexer_training_objective_is_preserved(loss_coeff):
    args = _args(dsa_indexer_loss_coeff=loss_coeff)
    normalize_dsa_args(args, _hf_config())
    assert args.dsa_indexer_loss_coeff == loss_coeff


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dsa_indexer_loss_coeff": -0.001}, "--dsa-indexer-loss-coeff must be non-negative"),
        ({"megatron_to_hf_mode": "bridge"}, "requires --megatron-to-hf-mode raw"),
        ({"spec": None}, "requires the shared DeepSeek-V3.2/GLM DSA spec"),
        ({"spec": ["miles_plugins.models.deepseek_v4", "get_dsv4_spec"]}, "requires the shared"),
        ({"context_parallel_size": 2, "allgather_cp": True}, "uses zigzag CP token partitioning"),
        ({"cp_comm_type": ["p2p"]}, "requires --cp-comm-type allgather"),
        ({"cp_comm_type": ["allgather", "a2a"]}, "requires --cp-comm-type allgather"),
    ],
    ids=[
        "negative-loss",
        "bridge",
        "no-spec",
        "v4-spec",
        "contiguous-cp",
        "explicit-p2p",
        "mixed-cp",
    ],
)
def test_incompatible_native_configuration_fails_early(overrides, message):
    with pytest.raises(ValueError, match=message):
        normalize_dsa_args(_args(**overrides), _hf_config())


def test_unsupported_checkpoint_cannot_select_native_dsa():
    with pytest.raises(ValueError, match="does not support model_type='deepseek_v4'"):
        normalize_dsa_args(_args(), _hf_config(model_type="deepseek_v4"))


def test_native_cp_preserves_per_layer_allgather_communication_with_zigzag_partitioning():
    args = _args(context_parallel_size=4, cp_comm_type=["allgather", "allgather"])
    normalize_dsa_args(args, _hf_config())
    assert args.cp_comm_type == ["allgather", "allgather"]
    assert args.allgather_cp is False


def test_native_dsa_keeps_the_requested_topk_backend():
    parsed = add_dsa_arguments(ArgumentParser()).parse_args(
        ["--dsa-impl", "megatron", "--miles-dsa-topk-backend", "flashinfer"]
    )
    args = _args(**vars(parsed))
    normalize_dsa_args(args, _hf_config())
    assert args.dsa_indexer_topk_backend == "flashinfer"


@pytest.fixture
def megatron_defaults(monkeypatch):
    # Exercise the real normalization entrypoint without importing GPU startup dependencies.
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.tokenizers.utils.build_tokenizer",
        SimpleNamespace(vocab_size_with_padding=lambda size, args: size),
    )
    monkeypatch.setitem(
        sys.modules, "megatron.training.arguments", SimpleNamespace(parse_args=None, validate_args=None)
    )
    monkeypatch.setitem(
        sys.modules, "miles.utils.hf_utils.config", SimpleNamespace(load_hf_config=lambda path: _hf_config())
    )
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/arguments.py"
    spec = importlib.util.spec_from_file_location("dsa_megatron_arguments_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.set_default_megatron_args


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["p2p"]),
        (["--cp-comm-type", "a2a"], ["a2a"]),
        (["--dsa-impl", "megatron"], ["allgather"]),
        (["--dsa-impl", "megatron", "--cp-comm-type", "allgather"], ["allgather"]),
    ],
    ids=["default-miles", "explicit-miles-cp", "default-native", "explicit-native-cp"],
)
def test_shared_parser_normalizes_omitted_cp_without_overriding_explicit_settings(megatron_defaults, argv, expected):
    parser = ArgumentParser()
    parser.add_argument("--cp-comm-type", nargs="+", default=["p2p"])
    add_dsa_arguments(parser)
    args = _args(
        **vars(parser.parse_args(argv)),
        optimizer="adam",
        fp16=False,
        seq_length=None,
        vocab_size=None,
        tokenizer_model=None,
        tokenizer_type=None,
        hf_checkpoint="/model",
    )
    megatron_defaults(args)
    assert args.cp_comm_type == expected


@pytest.mark.parametrize("argv,implementation", [([], "miles"), (["--dsa-impl", "megatron"], "megatron")])
def test_worker_parser_registers_dsa_arguments(argv, implementation):
    # Exercise the real registrar without importing the worker's GPU startup stack.
    path = Path(__file__).resolve().parents[4] / "miles/utils/debug_utils/run_megatron/worker/main.py"
    tree = ast.parse(path.read_text())
    registrar = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_register_worker_arguments"
    )
    namespace = {
        "argparse": argparse,
        "WORKER_SCRIPT_ARGS_BRIDGE": WORKER_SCRIPT_ARGS_BRIDGE,
        "add_dsv4_arguments": add_dsv4_arguments,
        "add_dsa_arguments": add_dsa_arguments,
    }
    exec(compile(ast.Module(body=[registrar], type_ignores=[]), str(path), "exec"), namespace)
    parser = ArgumentParser()
    parser.add_argument("--cp-comm-type", nargs="+", default=["p2p"])
    namespace["_register_worker_arguments"](parser)
    args = parser.parse_args(
        [
            "--script-hf-checkpoint",
            "/hf",
            "--script-token-ids-file",
            "/tokens.json",
            "--miles-dsa-topk-backend",
            "flashinfer",
        ]
        + argv
    )
    assert args.dsa_impl == implementation
    assert args.miles_dsa_topk_backend == "flashinfer"
    assert args.cp_comm_type is None
    assert WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(args).hf_checkpoint == Path("/hf")
