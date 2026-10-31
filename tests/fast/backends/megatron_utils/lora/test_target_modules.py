from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.lora.target_modules import resolve_megatron_lora_targets
from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets


class _Mapping(SimpleNamespace):
    def resolve(self, captures):
        def expand(name):
            for value in captures:
                name = name.replace("*", value, 1)
            return name

        hf = [self.hf_param] if isinstance(self.hf_param, str) else self.hf_param.values()
        return _mapping(expand(self.megatron_param), *(expand(name) for name in hf))


def _mapping(megatron, *hf):
    return _Mapping(megatron_param=megatron, hf_param=hf[0] if len(hf) == 1 else dict(enumerate(hf)))


def _resolve(targets, mappings, parameter_names, *, canonical=False, hf_mapping=None):
    return resolve_megatron_lora_targets(
        targets,
        mappings,
        parameter_names=set(parameter_names),
        hf_mapping=hf_mapping or HfWeightMapping(frozenset()),
        canonical=canonical,
    )


_QKV = _mapping(
    "decoder.layers.*.self_attention.linear_qkv.weight",
    *(f"model.layers.*.self_attn.{p}_proj.weight" for p in ("q", "k", "v")),
)


@pytest.mark.parametrize(
    "mtp_source",
    ["mtp.layers.0.self_attn.o_proj.weight", "model.layers.1.self_attn.o_proj.weight"],
    ids=["separate-namespace", "appended-layer"],
)
def test_scoped_attention_excludes_mtp(mtp_source):
    targets = resolve_hf_lora_targets({"model_type": "qwen3"}, target_modules=["attn"])
    output = _mapping("decoder.layers.*.self_attention.linear_proj.weight", "model.layers.*.self_attn.o_proj.weight")
    mtp = _mapping("mtp.layers.*.self_attention.linear_proj.weight", mtp_source)
    megatron_parameters = [
        "decoder.layers.0.self_attention.linear_qkv.weight",
        "decoder.layers.0.self_attention.linear_proj.weight",
        "mtp.layers.0.self_attention.linear_proj.weight",
    ]
    hf_mapping = HfWeightMapping(frozenset(f"model.layers.0.self_attn.{p}_proj.weight" for p in ("q", "k", "v", "o")))
    selected = _resolve(targets, [_QKV, output, mtp], megatron_parameters, hf_mapping=hf_mapping)
    assert set(selected) == {
        "decoder.layers.*.self_attention.linear_qkv",
        "decoder.layers.*.self_attention.linear_proj",
    }


def test_fused_selection_cannot_silently_expand():
    targets = ["model.layers.*.self_attn.q_proj"]
    with pytest.raises(AssertionError, match="requires all HF targets"):
        _resolve(targets, [_QKV], ["decoder.layers.0.self_attention.linear_qkv.weight"])
    adapter_targets = _resolve(targets, [_QKV], ["decoder.layers.0.self_attention.linear_qkv.weight"], canonical=True)
    assert adapter_targets == ["decoder.layers.*.self_attention.linear_q"]


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "sequential"])
def test_expert_representations_are_alternatives(grouped):
    target = "model.layers.*.mlp.experts.*.down_proj"
    mappings = [
        _mapping("decoder.layers.*.mlp.experts.linear_fc2.weight*", target + ".weight"),
        _mapping("decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight", target + ".weight"),
    ]
    module = "decoder.layers.0.mlp.experts." + ("linear_fc2" if grouped else "local_experts.0.linear_fc2")
    parameter = module + (".weight0" if grouped else ".weight")
    selected = _resolve([target], mappings, [parameter])
    assert len(selected) == 1
    assert ("local_experts" in next(iter(selected))) != grouped
    with pytest.raises(AssertionError, match="no Megatron modules"):
        _resolve([target], mappings, ["decoder.layers.0.mlp.linear_fc2.weight"])


def test_missing_hf_mapping_is_not_a_megatron_passthrough():
    with pytest.raises(AssertionError, match="no Megatron modules"):
        _resolve(
            ["model.layers.*.self_attn.unknown_proj"], [_QKV], ["decoder.layers.0.self_attention.linear_qkv.weight"]
        )


def test_one_to_one_mapping_keeps_bridge_module_name():
    target = "model.layers.*.self_attn.o_proj"
    mapping = _mapping("decoder.layers.*.self_attention.output_projection.weight", target + ".weight")
    adapter_targets = _resolve(
        [target], [mapping], ["decoder.layers.0.self_attention.output_projection.weight"], canonical=True
    )
    assert adapter_targets == ["decoder.layers.*.self_attention.output_projection"]


def test_absent_fused_alternative_does_not_reject_selection():
    mappings = [
        _mapping(
            "decoder.layers.*.self_attention.fused_qkv.weight",
            *(f"model.layers.*.self_attn.{p}_proj.weight" for p in ("q", "k", "v")),
        ),
        _mapping("decoder.layers.*.self_attention.linear_q.weight", "model.layers.*.self_attn.q_proj.weight"),
    ]
    selected = _resolve(["q_proj"], mappings, ["decoder.layers.0.self_attention.linear_q.weight"])
    assert set(selected) == {"decoder.layers.*.self_attention.linear_q"}


def test_canonical_selection_cannot_expand_across_layers():
    with pytest.raises(AssertionError, match="different projections"):
        _resolve(
            ["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.k_proj"],
            [_QKV],
            [f"decoder.layers.{layer}.self_attention.linear_qkv.weight" for layer in range(2)],
            canonical=True,
        )


def _gdn_in_proj(*hf_leaves):
    return _mapping(
        "decoder.layers.*.self_attention.in_proj.weight",
        *(f"model.layers.*.linear_attn.{leaf}.weight" for leaf in hf_leaves),
    )


_GDN_PARAMETER = ["decoder.layers.0.self_attention.in_proj.weight"]
_QWEN3_5_IN_PROJ = _gdn_in_proj("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def test_canonical_gdn_in_proj_adapts_the_selected_components():
    """Official Tinker trains in_proj_q/k/v/z with their own A and leaves b/a frozen."""
    targets = [f"model.layers.*.linear_attn.{leaf}" for leaf in ("in_proj_qkv", "in_proj_z")]
    assert set(_resolve(targets, [_QWEN3_5_IN_PROJ], _GDN_PARAMETER, canonical=True)) == {
        f"decoder.layers.*.self_attention.in_proj_{component}" for component in ("q", "k", "v", "z")
    }
    with pytest.raises(AssertionError, match="requires all HF targets"):
        _resolve(targets, [_QWEN3_5_IN_PROJ], _GDN_PARAMETER)


def test_canonical_gdn_in_proj_with_every_target_adapts_all_six_components():
    targets = [f"model.layers.*.linear_attn.{leaf}" for leaf in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
    assert set(_resolve(targets, [_QWEN3_5_IN_PROJ], _GDN_PARAMETER, canonical=True)) == {
        f"decoder.layers.*.self_attention.in_proj_{component}" for component in ("q", "k", "v", "z", "b", "a")
    }


def test_qwen3_next_in_proj_stays_one_fused_adapter():
    targets = [f"model.layers.*.linear_attn.{leaf}" for leaf in ("in_proj_qkvz", "in_proj_ba")]
    mapping = _gdn_in_proj("in_proj_qkvz", "in_proj_ba")
    assert _resolve(targets, [mapping], _GDN_PARAMETER, canonical=True) == ["decoder.layers.*.self_attention.in_proj"]


def _normalize(monkeypatch, targets, *, canonical):
    from miles.backends.megatron_utils.lora.target_modules import normalize_lora_targets_to_hf

    registry = SimpleNamespace(get_all_mappings=lambda: [_QWEN3_5_IN_PROJ])
    bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(config=None), _model_bridge=SimpleNamespace(mapping_registry=lambda: registry)
    )
    monkeypatch.setattr("miles.utils.megatron_bridge_utils.load_auto_bridge", lambda checkpoint: bridge)
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora.target_modules.HfWeightMapping.from_config",
        lambda config: HfWeightMapping(frozenset()),
    )
    return normalize_lora_targets_to_hf("/ckpt", targets, canonical=canonical, exclude_modules=[])


def test_a_gdn_component_selector_cannot_widen_to_its_whole_hf_module(monkeypatch):
    """in_proj_q alone would train in_proj_k and in_proj_v too, since HF holds q/k/v in one in_proj_qkv."""
    with pytest.raises(AssertionError, match=r"no Bridge mapping: \['in_proj_q'\]"):
        _normalize(monkeypatch, ["in_proj_q", "in_proj_z"], canonical=True)
    assert _normalize(monkeypatch, ["in_proj_[qkv]", "in_proj_z"], canonical=True) == [
        "model.layers.*.linear_attn.in_proj_qkv",
        "model.layers.*.linear_attn.in_proj_z",
    ]
