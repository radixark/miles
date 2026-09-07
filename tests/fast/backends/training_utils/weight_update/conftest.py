import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import pytest

_P2P_TRANSFER_UTILS_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils"
_P2P_CELL_UPDATER_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_inference_cell_updater"
_P2P_PROTOCOL_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p"

_P2P_PROTOCOL_EXTERNAL_SDKS = {
    "mooncake.engine": {"TransferEngine": object},
    "sglang.srt.server_args": {"ServerArgs": object},
    "sglang.srt.configs.device_config": {"DeviceConfig": object},
    "sglang.srt.configs.load_config": {"LoadConfig": object},
    "sglang.srt.configs.model_config": {"ModelConfig": object},
    "sglang.srt.distributed.parallel_state": {"ParallelismContext": object, "RankParallelismConfig": object},
    "sglang.srt.layers.moe": {"initialize_moe_config": object},
    "sglang.srt.layers.quantization.fp4_utils": {"initialize_fp4_gemm_config": object},
    "sglang.srt.layers.quantization.fp8_utils": {"initialize_fp8_gemm_config": object},
    "sglang.srt.model_loader": {"get_model": object},
    "sglang.srt.model_loader.loader": {"post_load_weights": object},
    "sglang.srt.model_loader.parameter_mapper": {"ParameterMapper": object},
}


@contextmanager
def stubbed_missing_external_sdks(module_attributes: dict[str, dict[str, object]]) -> Iterator[None]:
    created_modules: list[str] = []
    created_attributes: list[tuple[ModuleType, str]] = []

    for module_name, attributes in module_attributes.items():
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            if name in sys.modules:
                continue
            try:
                importlib.import_module(name)
                continue
            except ImportError:
                pass
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            created_modules.append(name)
            if depth > 1:
                parent = sys.modules[".".join(parts[: depth - 1])]
                setattr(parent, parts[depth - 1], module)
                created_attributes.append((parent, parts[depth - 1]))
        for attribute, value in attributes.items():
            module = sys.modules[module_name]
            if not hasattr(module, attribute):
                setattr(module, attribute, value)
                created_attributes.append((module, attribute))

    try:
        yield
    finally:
        for parent, attribute in reversed(created_attributes):
            delattr(parent, attribute)
        for name in reversed(created_modules):
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def p2p_transfer_utils() -> ModuleType:
    with stubbed_missing_external_sdks(
        {
            "mooncake.engine": {"TransferEngine": object},
            "sglang.srt.server_args": {"ServerArgs": object},
        }
    ):
        return importlib.import_module(_P2P_TRANSFER_UTILS_MODULE)


@pytest.fixture(scope="module")
def p2p_inference_cell_updater() -> ModuleType:
    with stubbed_missing_external_sdks(
        {
            "mooncake.engine": {"TransferEngine": object},
            "sglang.srt.server_args": {"ServerArgs": object},
        }
    ):
        return importlib.import_module(_P2P_CELL_UPDATER_MODULE)


@pytest.fixture(scope="module")
def p2p_protocol() -> Iterator[ModuleType]:
    with stubbed_missing_external_sdks(_P2P_PROTOCOL_EXTERNAL_SDKS):
        yield importlib.import_module(_P2P_PROTOCOL_MODULE)
