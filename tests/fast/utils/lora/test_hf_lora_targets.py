"""LoRA selection uses model defaults and preserves explicit user choices."""

import pytest

from miles.utils.lora.hf_lora_targets import (
    exclude_hf_lora_targets,
    expand_packed_hf_lora_targets,
    parse_lora_targets,
    resolve_hf_lora_targets,
)


@pytest.mark.parametrize("selection", [None, ["all-linear"], ["attn", "mlp"], ["attn", "mlp", "attn"]])
def test_missing_targets_and_all_linear_use_model_defaults(selection):
    config = {"model_type": "qwen3"}
    targets = resolve_hf_lora_targets(config, target_modules=selection)
    assert targets == resolve_hf_lora_targets(config, target_modules=["all-linear"])
    assert "model.layers.*.self_attn.q_proj" in targets
    assert "model.layers.*.mlp.gate_proj" in targets
    assert "lm_head" not in targets


@pytest.mark.parametrize(
    "config,value,expected",
    [
        ({"model_type": "custom"}, "q_proj, k_proj", ["q_proj", "k_proj"]),
        ({"model_type": "custom"}, ["q_proj", "k_proj"], ["q_proj", "k_proj"]),
        ({"model_type": "qwen3"}, "unembed,lm_head", ["lm_head"]),
        (
            {"model_type": "qwen3"},
            "attn, lm_head",
            [f"model.layers.*.self_attn.{name}_proj" for name in ("q", "k", "v", "o")] + ["lm_head"],
        ),
        (
            {"model_type": "gpt_oss"},
            "mlp",
            ["model.layers.*.mlp.experts.gate_up_proj", "model.layers.*.mlp.experts.down_proj"],
        ),
        ({"model_type": "custom"}, "model.mlp", ["model.mlp"]),
    ],
)
def test_explicit_targets_preserve_user_choices(config, value, expected):
    assert resolve_hf_lora_targets(config, target_modules=parse_lora_targets(value)) == expected


def test_exclude_leaf_applies_to_scoped_model_defaults():
    targets = resolve_hf_lora_targets({"model_type": "qwen3"})
    selected = exclude_hf_lora_targets(targets, parse_lora_targets("o_proj, down_proj"))
    assert "model.layers.*.self_attn.o_proj" not in selected
    assert "model.layers.*.mlp.down_proj" not in selected
    assert "model.layers.*.self_attn.q_proj" in selected


def test_exclude_all_rejects_empty_selection():
    with pytest.raises(AssertionError, match="empty after"):
        exclude_hf_lora_targets(["q_proj", "k_proj"], ["q_proj", "k_proj"])


def test_nonexistent_exclusion_does_not_change_selection():
    assert exclude_hf_lora_targets(["q_proj", "k_proj"], ["nonexistent"]) == ["q_proj", "k_proj"]


def test_empty_selector_is_rejected():
    with pytest.raises(AssertionError, match="empty entries"):
        parse_lora_targets("q_proj,,k_proj")


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("scoped", [True, False])
def test_paired_targets_include_packed_experts(dense, scoped, caplog):
    modules = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj"]
    if dense:
        modules += [f"model.layers.0.mlp.{name}_proj" for name in ("gate", "up", "down")]
    prefix = "model.layers.*.mlp.experts." if scoped else ""
    targets = [prefix + name + "_proj" for name in ("gate", "up", "down")]
    expected = targets if dense and not scoped else [prefix + "down_proj"]
    assert expand_packed_hf_lora_targets(targets, modules) == expected + [prefix + "gate_up_proj"]
    assert "Expanding paired gate_proj/up_proj" in caplog.text
    assert expand_packed_hf_lora_targets(targets[:1], modules) == targets[:1]
    assert expand_packed_hf_lora_targets(targets, ["model.layers.0.mlp.gate_proj"]) == targets
