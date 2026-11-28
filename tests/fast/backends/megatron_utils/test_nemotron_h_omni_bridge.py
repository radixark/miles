import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class Mapping:
    def __init__(self, megatron_param: str, hf_param) -> None:
        self.megatron_param = megatron_param
        self.hf_param = hf_param


class QKVMapping(Mapping):
    pass


class Registry:
    def __init__(self, *mappings: Mapping) -> None:
        self.mappings = list(mappings)


BASE_MAPPINGS = [
    Mapping("embedding.word_embeddings.weight", "backbone.embeddings.weight"),
    Mapping(
        "decoder.layers.*.mlp.experts.linear_fc1.weight*",
        "backbone.layers.*.mixer.experts.*.up_proj.weight",
    ),
    QKVMapping(
        "decoder.layers.*.self_attention.linear_qkv.weight",
        {
            "q": "backbone.layers.*.mixer.q_proj.weight",
            "k": "backbone.layers.*.mixer.k_proj.weight",
            "v": "backbone.layers.*.mixer.v_proj.weight",
        },
    ),
    Mapping("mtp.layers.*.enorm.weight", "mtp.layers.*.enorm.weight"),
]


class FakeNemotronHBridge:
    def provider_bridge(self, hf_pretrained):
        return SimpleNamespace(config=hf_pretrained.config, state=hf_pretrained.state)

    def mapping_registry(self):
        self.seen_hf_config = self.hf_config
        return Registry(*BASE_MAPPINGS)


def load_module(monkeypatch: pytest.MonkeyPatch, supported: tuple[str, ...]):
    registrations = []

    def register_bridge(**kwargs):
        registrations.append(kwargs)
        return lambda cls: cls

    auto_bridge = ModuleType("megatron.bridge.models.conversion.auto_bridge")
    auto_bridge.SUPPORTED_HF_ARCHITECTURES = supported
    auto_bridge.SUPPORTED_HF_ARCHITECTURES_DISPLAY = ""
    conversion = ModuleType("megatron.bridge.models.conversion")
    conversion.auto_bridge = auto_bridge
    modules = {
        "megatron.bridge.models.conversion": conversion,
        "megatron.bridge.models.conversion.auto_bridge": auto_bridge,
        "megatron.bridge.models.conversion.mapping_registry": ModuleType("mapping_registry"),
        "megatron.bridge.models.conversion.model_bridge": ModuleType("model_bridge"),
        "megatron.core.models.mamba": ModuleType("mamba"),
        "miles_plugins.megatron_bridge.nemotron_h": ModuleType("nemotron_h"),
    }
    modules["megatron.bridge.models.conversion.mapping_registry"].MegatronMappingRegistry = Registry
    modules["megatron.bridge.models.conversion.model_bridge"].MegatronModelBridge = SimpleNamespace(
        register_bridge=register_bridge
    )
    modules["megatron.core.models.mamba"].MambaModel = object
    modules["miles_plugins.megatron_bridge.nemotron_h"].bridge_class = lambda: FakeNemotronHBridge
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[4] / "miles_plugins/megatron_bridge/nemotron_h_omni.py"
    spec = importlib.util.spec_from_file_location("nemotron_h_omni_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, auto_bridge, registrations


def omni_config():
    llm_config = SimpleNamespace(
        hidden_size=4096,
        layers_block_type=["mamba", "moe", "attention", "mlp"],
        mtp_layers_block_type=["attention", "moe"],
        num_nextn_predict_layers=1,
    )
    return SimpleNamespace(
        architectures=["NemotronH_Omni_Reasoning_V3"],
        llm_config=llm_config,
        vision_config={},
    )


def test_language_model_config_derives_legacy_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = load_module(monkeypatch, ("ForCausalLM",))
    view = module.language_model_config(omni_config())
    assert view.hidden_size == 4096
    assert view.hybrid_override_pattern == "ME*-"
    assert view.mtp_hybrid_override_pattern == "*E"
    assert not hasattr(view, "vision_config")

    explicit = SimpleNamespace(llm_config=SimpleNamespace(hybrid_override_pattern="M*", layers_block_type=["mamba"]))
    assert module.language_model_config(explicit).hybrid_override_pattern == "M*"

    text_config = SimpleNamespace(hidden_size=1)
    assert module.language_model_config(text_config) is text_config


def test_bridge_registers_the_omni_architecture_and_admits_it(monkeypatch: pytest.MonkeyPatch) -> None:
    module, auto_bridge, registrations = load_module(monkeypatch, ("ForCausalLM",))
    bridge_cls = module.bridge_class()
    assert module.bridge_class() is bridge_cls
    assert [r["source"] for r in registrations] == ["NemotronH_Omni_Reasoning_V3"]
    assert auto_bridge.SUPPORTED_HF_ARCHITECTURES == ("ForCausalLM", "NemotronH_Omni_Reasoning_V3")
    assert "'NemotronH_Omni_Reasoning_V3'" in auto_bridge.SUPPORTED_HF_ARCHITECTURES_DISPLAY
    module._allow_omni_architecture()
    assert auto_bridge.SUPPORTED_HF_ARCHITECTURES.count("NemotronH_Omni_Reasoning_V3") == 1


def test_provider_sees_the_language_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = load_module(monkeypatch, ("ForCausalLM",))
    hf_pretrained = SimpleNamespace(
        config=omni_config(),
        state={"language_model.backbone.embeddings.weight": 1},
    )
    provider = module.bridge_class()().provider_bridge(hf_pretrained)
    assert provider.config.hidden_size == 4096
    assert provider.config.hybrid_override_pattern == "ME*-"
    assert provider.state is hf_pretrained.state


def test_mapping_registry_prefixes_language_model_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = load_module(monkeypatch, ("ForCausalLM",))
    bridge = module.bridge_class()()
    original_config = omni_config()
    bridge.hf_config = original_config
    registry = bridge.mapping_registry()

    assert bridge.seen_hf_config.hybrid_override_pattern == "ME*-"
    assert bridge.hf_config is original_config
    assert [m.megatron_param for m in registry.mappings] == [m.megatron_param for m in BASE_MAPPINGS]
    assert [type(m) for m in registry.mappings] == [type(m) for m in BASE_MAPPINGS]
    by_name = {m.megatron_param: m for m in registry.mappings}
    assert by_name["embedding.word_embeddings.weight"].hf_param == "language_model.backbone.embeddings.weight"
    assert (
        by_name["decoder.layers.*.mlp.experts.linear_fc1.weight*"].hf_param
        == "language_model.backbone.layers.*.mixer.experts.*.up_proj.weight"
    )
    assert by_name["decoder.layers.*.self_attention.linear_qkv.weight"].hf_param == {
        "q": "language_model.backbone.layers.*.mixer.q_proj.weight",
        "k": "language_model.backbone.layers.*.mixer.k_proj.weight",
        "v": "language_model.backbone.layers.*.mixer.v_proj.weight",
    }
    assert by_name["mtp.layers.*.enorm.weight"].hf_param == "language_model.mtp.layers.*.enorm.weight"
    assert BASE_MAPPINGS[0].hf_param == "backbone.embeddings.weight"


def test_registry_mapping_fallback_supports_legacy_private_name(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = load_module(monkeypatch, ("ForCausalLM",))
    mappings = [BASE_MAPPINGS[0]]
    assert module._registry_mappings(SimpleNamespace(_mappings=mappings)) is mappings
