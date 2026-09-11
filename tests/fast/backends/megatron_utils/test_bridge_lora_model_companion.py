import sys
import types

import pytest
import torch

_MISSING = object()


@pytest.fixture(scope="module")
def helpers_module():
    installed: dict[str, object] = {}

    def install(name: str, module) -> None:
        installed[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = module

    def stub(name: str, attrs: dict | None = None, is_package: bool = False):
        module = types.ModuleType(name)
        if is_package:
            module.__path__ = []
        for attr_name, value in (attrs or {}).items():
            setattr(module, attr_name, value)
        install(name, module)
        return module

    try:
        stub("megatron", is_package=True)
        stub("megatron.core", is_package=True)
        stub("megatron.core.utils", {"get_attr_wrapped_model": lambda *a, **k: None})
        install("megatron.core.parallel_state", types.SimpleNamespace())

        from miles.backends.megatron_utils import bridge_lora_helpers

        yield bridge_lora_helpers
    finally:
        for name, previous in installed.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture
def fake_parallel_state(monkeypatch: pytest.MonkeyPatch) -> None:
    parallel = types.SimpleNamespace(
        pp=types.SimpleNamespace(rank=0),
        tp=types.SimpleNamespace(rank=1),
        cp=types.SimpleNamespace(rank=2),
        intra_dp=types.SimpleNamespace(rank=3),
    )
    monkeypatch.setattr("miles.backends.training_utils.parallel.get_parallel_state", lambda: parallel)


def test_every_bridge_lora_chunk_receives_a_model_companion(helpers_module, fake_parallel_state: None) -> None:
    """Every bridge LoRA chunk carries its own companion for sample bookkeeping."""
    chunks = [torch.nn.Module(), torch.nn.Module()]

    helpers_module._install_model_companions(chunks)

    assert [chunk.model_companion.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.model_companion.snapshot_sample_consumptions(is_skipped=False) == {} for chunk in chunks)
