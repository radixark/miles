"""Exports read back to the same HF-keyed tensors; conversions refuse what would change the training."""

import json

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter, write_sampler_adapter
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization_conversion import (
    ConversionOp,
    apply_parametrization_conversion,
    assert_parametrization_matches_run,
    plan_parametrization_conversion,
)
from miles.backends.training_utils.checkpoint.adapter.readers import (
    AdapterSourceKind,
    MilesAdapterDirReader,
    PeftAdapterReader,
    classify_adapter_source,
)
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

L = "model.layers.0"
QKV = (f"{L}.linear_attn.in_proj_q", f"{L}.linear_attn.in_proj_k", f"{L}.linear_attn.in_proj_v")
EXPERTS = (f"{L}.mlp.experts.gate_proj", f"{L}.mlp.experts.up_proj", f"{L}.mlp.experts.down_proj")
QWEN3 = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})
# GDN v has more rows than q / k, as in Qwen3.5 (more value heads than key heads)
QWEN35 = get_hf_lora_targets(
    {
        "model_type": "qwen3_5_text",
        "layer_types": ["linear_attention"],
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 2,
    }
)
QKV_ROWS = (4, 4, 8)


def _run(**overrides):
    fields = dict(
        base_model="m",
        lora_rank=2,
        lora_alpha=4.0,
        lora_type="canonical_lora",
        target_modules=(*QKV, *EXPERTS),
        experts_shared_outer_loras=False,
    )
    return AdapterParametrization(**(fields | overrides))


def _weights(shared_qkv: bool):
    g = torch.Generator().manual_seed(0)
    w = {}
    for i, (m, rows) in enumerate(zip(QKV, QKV_ROWS, strict=True)):
        if not shared_qkv or i == 0:
            w[f"{m}.lora_A"] = torch.randn(2, 8, generator=g)
        w[f"{m}.lora_B"] = torch.randn(rows, 2, generator=g)
    for m in EXPERTS:
        w[f"{m}.lora_A"] = torch.randn(3, 2, 8, generator=g)
        w[f"{m}.lora_B"] = torch.randn(3, 8, 2, generator=g)
    return w


@pytest.mark.parametrize("shared_qkv", [False, True], ids=["canonical", "fused_gdn"])
def test_peft_export_reads_back_bitwise(tmp_path, shared_qkv):
    run = _run(shared_lora_a_groups=(QKV,) if shared_qkv else ())
    weights = {k: v.to(torch.bfloat16) for k, v in _weights(shared_qkv).items()}
    write_peft_adapter(tmp_path, weights, run)
    names = json.loads((tmp_path / "adapter_config.json").read_text())["target_modules"]
    assert any("in_proj_qkv" in n for n in names) == shared_qkv, "shared-A GDN components go back to their HF name"
    assert classify_adapter_source(tmp_path, QWEN35, load_optimizer=False) == AdapterSourceKind.PEFT
    reader = PeftAdapterReader(tmp_path, run, QWEN35)
    assert reader.read_weights().keys() == weights.keys()
    assert all(torch.equal(reader.read_weights()[k], v) for k, v in weights.items())
    assert reader.source.parametrization.shared_lora_a_groups == run.shared_lora_a_groups


def test_sampler_export_reads_back_through_the_adapter_dir_reader(tmp_path):
    run = _run(shared_lora_a_groups=(QKV,))
    weights = {k: v.to(torch.bfloat16) for k, v in _weights(True).items()}
    write_sampler_adapter(tmp_path, weights, run)
    write_parametrization(tmp_path, run)
    reader = MilesAdapterDirReader(tmp_path, QWEN35)
    assert not reader.source.has_optimizer_state
    assert all(torch.equal(reader.read_weights()[k], v) for k, v in weights.items())


def test_splitting_a_shared_a_is_weights_only():
    source, target = _run(shared_lora_a_groups=(QKV,)), _run()
    steps = plan_parametrization_conversion(source, target, source_dtype=torch.float32)
    assert [s.op for s in steps] == [ConversionOp.SPLIT_SHARED_LORA_A]
    assert_parametrization_matches_run(source, target, steps, load_optimizer=False, has_optimizer_state=True)
    with pytest.raises(AssertionError, match="needs the source's own parametrization"):
        assert_parametrization_matches_run(source, target, steps, load_optimizer=True, has_optimizer_state=True)
    split = apply_parametrization_conversion(_weights(True), steps)
    assert torch.equal(split[f"{QKV[1]}.lora_A"], split[f"{QKV[0]}.lora_A"])


def test_merging_requires_equal_copies():
    steps = plan_parametrization_conversion(_run(), _run(shared_lora_a_groups=(QKV,)), source_dtype=torch.float32)
    with pytest.raises(AssertionError, match="cannot share A"):
        apply_parametrization_conversion(_weights(False), steps)


def test_shared_outer_broadcasts_for_weights_and_collapses_only_if_equal():
    source, target = _run(experts_shared_outer_loras=True), _run()
    steps = plan_parametrization_conversion(source, target, source_dtype=torch.float32)
    w = _weights(False)
    for key in (f"{EXPERTS[0]}.lora_A", f"{EXPERTS[1]}.lora_A", f"{EXPERTS[2]}.lora_B"):
        w[key] = w[key][:1]
    out = apply_parametrization_conversion(w, steps, num_experts=3)
    assert out[f"{EXPERTS[0]}.lora_A"].shape[0] == 3
    back = plan_parametrization_conversion(target, source, source_dtype=torch.float32)
    with pytest.raises(AssertionError, match="experts differ"):
        apply_parametrization_conversion(_weights(False), back)


@pytest.mark.parametrize("field, value", [("base_model", "other"), ("lora_rank", 4)])
def test_mismatches_are_refused(field, value):
    with pytest.raises(AssertionError):
        assert_parametrization_matches_run(
            _run(**{field: value}), _run(), [], load_optimizer=False, has_optimizer_state=False
        )


def test_peft_features_that_change_tensor_meaning_are_refused(tmp_path):
    save_file(
        {"base_model.model.x.lora_A.weight": torch.zeros(2, 2), "base_model.model.x.lora_B.weight": torch.zeros(2, 2)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "use_dora": True}))
    with pytest.raises(AssertionError, match="use_dora"):
        PeftAdapterReader(tmp_path, _run(), QWEN3)


def test_tinker_names_classify_as_external(tmp_path):
    save_file(
        {"base_model.model.model.unembed_tokens.lora_A.weight": torch.zeros(2, 2)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    (tmp_path / "adapter_config.json").write_text("{}")
    assert classify_adapter_source(tmp_path, QWEN3, load_optimizer=False) == AdapterSourceKind.EXTERNAL_WEIGHTS


def test_a_head_adapter_reads_back(tmp_path):
    run = _run(target_modules=("lm_head",))
    weights = {
        "lm_head.lora_A": torch.ones(2, 8, dtype=torch.bfloat16),
        "lm_head.lora_B": torch.ones(16, 2, dtype=torch.bfloat16),
    }
    write_peft_adapter(tmp_path, weights, run)
    assert PeftAdapterReader(tmp_path, run, QWEN3).read_weights().keys() == weights.keys()
