from types import SimpleNamespace

import pytest
import torch

from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles.utils.lora.utils import get_adapter_target_modules
from miles_plugins.models.inkling.lora import _export_dense_mlp, _export_experts, resolve_inkling_adapter_targets


class _LocalGather:
    def add(self, group, tensor, dim):
        return SimpleNamespace(get=lambda: tensor)


@pytest.mark.parametrize("multimodal", [False, True], ids=["text", "multimodal"])
def test_hf_mlp_selection_matches_existing_native_export(multimodal):
    config = dict(model_type="inkling_text", mlp_layer_types=["dense", "sparse"], n_shared_experts=0)
    if multimodal:
        config = dict(model_type="inkling_mm_model", text_config=config)
    hf_targets = resolve_hf_lora_targets(config)
    assert resolve_inkling_adapter_targets(config, hf_targets) == "all-linear"
    prefix = "model.language_model" if multimodal else "model"
    assert f"{prefix}.layers.*.mlp.gate_proj" in hf_targets
    assert f"{prefix}.layers.*.mlp.experts.gate_up_proj" in hf_targets
    tensor = torch.ones(2, 2)
    dense = SimpleNamespace(
        hf_prefix="language_model.layers.0.mlp.",
        load_meta={"i_loc": 1},
        fc1_A=tensor,
        fc1_B=tensor,
        fc2_A=tensor,
        fc2_B=tensor,
    )
    experts = SimpleNamespace(
        hf_prefix="language_model.layers.1.mlp.experts.",
        **{f"w{projection}_{factor}": tensor for projection in (1, 2, 3) for factor in ("A", "B")},
    )
    plan = _export_dense_mlp(dense, _LocalGather()) + _export_experts(experts, _LocalGather())
    weights = {name: value() if callable(value) else value for name, value in plan}
    assert "language_model.layers.0.mlp.gate_up_proj" in get_adapter_target_modules(weights)
    assert weights["language_model.layers.0.mlp.gate_up_proj.lora_A.weight"] is tensor
    assert torch.equal(weights["language_model.layers.0.mlp.gate_up_proj.lora_B.weight"], tensor)
    with pytest.raises(AssertionError, match="complete adapter layout"):
        resolve_inkling_adapter_targets(config, [target for target in hf_targets if not target.endswith(".up_proj")])


def test_legacy_config_selects_the_same_native_adapters():
    legacy = dict(model_type="inkling_model", dense_mlp_idx=1, num_hidden_layers=2, n_shared_experts=1)
    native = dict(model_type="inkling_text", mlp_layer_types=["dense", "sparse"], n_shared_experts=1)
    assert resolve_hf_lora_targets(legacy) == resolve_hf_lora_targets(native)
    resolve_inkling_adapter_targets(legacy, resolve_hf_lora_targets(legacy))
