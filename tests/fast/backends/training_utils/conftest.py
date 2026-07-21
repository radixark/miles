import importlib
import sys
import types
from enum import Enum
from types import SimpleNamespace

import pytest

_MISSING = object()
_MILES_MODULES = [
    "miles.backends.sglang_utils.arguments",
    "miles.backends.training_utils.cp_utils",
    "miles.backends.training_utils.log_utils",
    "miles.backends.training_utils.loss",
    "miles.backends.training_utils.loss_hub.losses",
    "miles.backends.training_utils.parallel",
    "miles.ray.rollout.train_data_conversion",
    "miles.utils.arguments",
    "miles.utils.types",
]


def _install_import_stubs(monkeypatch):
    for name in ["sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai"]:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
    protocol.Tool = object
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.openai.protocol", protocol)

    server_args = types.ModuleType("sglang.srt.server_args")

    class _ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            return None

    server_args.ServerArgs = _ServerArgs
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    chat_template_utils = types.ModuleType("miles.utils.chat_template_utils")
    chat_template_utils.__path__ = []
    chat_template_utils.resolve_fixed_chat_template = lambda *args, **kwargs: (None, {})
    monkeypatch.setitem(sys.modules, "miles.utils.chat_template_utils", chat_template_utils)

    tito_tokenizer = types.ModuleType("miles.utils.chat_template_utils.tito_tokenizer")

    class _TITOTokenizerType(Enum):
        DEFAULT = "default"

    tito_tokenizer.TITOTokenizerType = _TITOTokenizerType
    monkeypatch.setitem(sys.modules, "miles.utils.chat_template_utils.tito_tokenizer", tito_tokenizer)

    sglang_router = types.ModuleType("sglang_router")
    sglang_router.__path__ = []
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)

    launch_router = types.ModuleType("sglang_router.launch_router")

    class _RouterArgs:
        @staticmethod
        def add_cli_args(parser, *args, **kwargs):
            return None

    launch_router.RouterArgs = _RouterArgs
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)


@pytest.fixture
def miles(monkeypatch):
    previous_modules = {name: sys.modules.get(name, _MISSING) for name in _MILES_MODULES}
    for name in _MILES_MODULES:
        sys.modules.pop(name, None)

    _install_import_stubs(monkeypatch)

    cp_utils = importlib.import_module("miles.backends.training_utils.cp_utils")
    log_utils = importlib.import_module("miles.backends.training_utils.log_utils")
    loss = importlib.import_module("miles.backends.training_utils.loss")
    losses = importlib.import_module("miles.backends.training_utils.loss_hub.losses")
    parallel = importlib.import_module("miles.backends.training_utils.parallel")
    train_data_conversion = importlib.import_module("miles.ray.rollout.train_data_conversion")
    arguments = importlib.import_module("miles.utils.arguments")
    types_module = importlib.import_module("miles.utils.types")

    try:
        yield SimpleNamespace(
            arguments=arguments,
            convert_samples_to_train_data=train_data_conversion.convert_samples_to_train_data,
            split_train_data_by_dp=train_data_conversion.split_train_data_by_dp,
            train_data_conversion=train_data_conversion,
            cp_utils=cp_utils,
            log_utils=log_utils,
            loss=loss,
            get_pg_loss_reducer=losses.get_pg_loss_reducer,
            get_sum_of_sample_mean=cp_utils.get_sum_of_sample_mean,
            GroupInfo=parallel.GroupInfo,
            ParallelState=parallel.ParallelState,
            Sample=types_module.Sample,
        )
    finally:
        for name in reversed(_MILES_MODULES):
            previous = previous_modules[name]
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
