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
