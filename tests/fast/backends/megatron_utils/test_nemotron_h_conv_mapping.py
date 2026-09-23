import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class Mapping:
    def __init__(self, megatron_param: str, hf_param: str) -> None:
        self.megatron_param = megatron_param
        self.hf_param = hf_param


class MambaMapping(Mapping):
    pass


def load_registry(monkeypatch: pytest.MonkeyPatch, base: list[Mapping], attribute: str) -> list[Mapping]:
    class Registry:
        def __init__(self, *mappings: Mapping) -> None:
            setattr(self, attribute, list(mappings))

    modules = {
        "megatron.bridge.models.conversion.mapping_registry": {"MegatronMappingRegistry": Registry},
        "megatron.bridge.models.conversion.model_bridge": {
            "MegatronModelBridge": SimpleNamespace(register_bridge=lambda **kwargs: lambda cls: cls)
        },
        "megatron.bridge.models.conversion.param_mapping": {
            "AutoMapping": Mapping,
            "MambaConv1dMapping": MambaMapping,
        },
        "megatron.bridge.models.nemotronh.nemotron_h_bridge": {
            "NemotronHBridge": type("NemotronHBridge", (), {"mapping_registry": lambda self: Registry(*base)})
        },
        "megatron.core.models.mamba": {"MambaModel": object},
    }
    for name, attributes in modules.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    path = Path(__file__).resolve().parents[4] / "miles_plugins/megatron_bridge/nemotron_h.py"
    spec = importlib.util.spec_from_file_location("nemotron_h_mapping_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = module._build_bridge_subclass()().mapping_registry()
    return getattr(registry, attribute)


@pytest.mark.parametrize("attribute", ["mappings", "_mappings"])
@pytest.mark.parametrize("existing_suffixes", [(), ("weight",), ("weight", "bias")])
def test_convolution_mapping_coverage(
    monkeypatch: pytest.MonkeyPatch, attribute: str, existing_suffixes: tuple[str, ...]
) -> None:
    legacy = [
        MambaMapping(f"decoder.layers.*.mixer.conv1d.{suffix}", f"backbone.layers.*.mixer.conv1d.{suffix}")
        for suffix in ("weight", "bias")
    ]
    existing = [
        MambaMapping(f"decoder.layers.*.mixer.conv1d_{suffix}", f"backbone.layers.*.mixer.conv1d.{suffix}")
        for suffix in existing_suffixes
    ]
    base = legacy + existing
    original_names = [mapping.megatron_param for mapping in base]

    mappings = load_registry(monkeypatch, base, attribute)
    by_name = {mapping.megatron_param: mapping for mapping in mappings}
    assert len(by_name) == len(mappings)
    for separator in (".", "_"):
        for suffix in ("weight", "bias"):
            mapping = by_name[f"decoder.layers.*.mixer.conv1d{separator}{suffix}"]
            assert isinstance(mapping, MambaMapping)
            assert mapping.hf_param == f"backbone.layers.*.mixer.conv1d.{suffix}"
    for mapping in base:
        assert by_name[mapping.megatron_param] is mapping
    assert [mapping.megatron_param for mapping in base] == original_names
    assert "decoder.layers.*.mlp.router.weight" in by_name
