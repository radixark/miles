"""LoRA selection uses model defaults and preserves explicit user choices."""

import pytest

from miles.utils.hf_lora_targets import (
    exclude_hf_lora_targets,
    parse_lora_targets,
    resolve_hf_lora_targets,
)


def test_missing_targets_and_all_linear_use_model_defaults():
    config = {"model_type": "qwen3"}
    targets = resolve_hf_lora_targets(config)
    assert targets == resolve_hf_lora_targets(config, target_modules=["all-linear"])
    assert "model.layers.*.self_attn.q_proj" in targets
    assert "model.layers.*.mlp.gate_proj" in targets
    assert "lm_head" not in targets


@pytest.mark.parametrize(
    "value",
    ["q_proj, k_proj", "q_proj,k_proj", ["q_proj", "k_proj"]],
    ids=["spaced", "comma-separated", "list"],
)
def test_explicit_targets_override_group_flags(value):
    targets = resolve_hf_lora_targets(
        {"model_type": "custom"},
        target_modules=parse_lora_targets(value),
        train_attn=False,
        train_mlp=True,
        train_unembed=True,
    )
    assert targets == ["q_proj", "k_proj"]


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
