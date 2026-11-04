import argparse
import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from miles.utils.args.runtime import TrainerConfig


@pytest.fixture
def raw_mbridge_loader(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], list[set[str]]]:
    def configure(fail: bool) -> list[set[str]]:
        observed: list[set[str]] = []

        def load_weights(models: list[torch.nn.Module], path: str, *, memory_efficient: bool) -> None:
            for model in models:
                observed.append(set(model.state_dict()))
                model.weight.data.fill_(5)
            if fail:
                raise RuntimeError("HF loading failed")

        bridge = Mock(load_weights=load_weights)
        mbridge = ModuleType("mbridge")
        mbridge.AutoBridge = Mock(from_pretrained=Mock(return_value=bridge))
        monkeypatch.setitem(sys.modules, "mbridge", mbridge)
        monkeypatch.setitem(sys.modules, "miles_plugins.mbridge", ModuleType("miles_plugins.mbridge"))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        return observed

    return configure


@pytest.fixture(scope="module")
def apply_bridge_runtime_config() -> Callable:
    # Execute the production helper without importing GPU-only Megatron modules.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/model_provider.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_bridge_runtime_config"
    )
    namespace = {"argparse": argparse, "TrainerConfig": TrainerConfig}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_apply_bridge_runtime_config"]


@pytest.fixture
def runtime_args() -> argparse.Namespace:
    backend = argparse.Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
        attention_softmax_in_fp32=False,
        gradient_accumulation_fusion=False,
        fp32_residual_connection=False,
        deterministic_mode=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=[],
        cpu_offloading_num_layers=0,
        distribute_saved_activations=False,
        tp_comm_overlap=False,
        fp8=None,
        fp8_recipe=None,
        attention_backend="auto",
        mtp_num_layers=1,
        mtp_detach_heads=True,
        moe_token_dispatcher_type="alltoall",
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        moe_router_bias_update_rate=None,
        moe_aux_loss_coeff=None,
    )
    return argparse.Namespace(backend=backend, dsa_attention_backend=None, enable_mtp_training=False)


@pytest.fixture
def bridge_sdk(monkeypatch: pytest.MonkeyPatch) -> Mock:
    sdk = Mock()
    module = ModuleType("megatron.bridge")
    module.AutoBridge = sdk
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return sdk


@pytest.fixture
def transformer_engine_quantizer(monkeypatch: pytest.MonkeyPatch) -> Mock:
    sdk = Mock()
    engine = ModuleType("transformer_engine")
    engine.pytorch = ModuleType("transformer_engine.pytorch")
    engine.pytorch.MXFP8Quantizer = sdk
    constants = ModuleType("transformer_engine.pytorch.constants")
    constants.TE_DType = {torch.float8_e4m3fn: object()}
    engine.pytorch.constants = constants
    for module in (engine, engine.pytorch, constants):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return sdk
