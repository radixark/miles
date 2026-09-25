"""Run in the Miles training image: pytest tests/manual/models/test_lora_target_resolution.py -v."""

from types import SimpleNamespace

import pytest
import torch
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.bridge.models.qwen.qwen3_next_bridge import Qwen3NextBridge
from transformers import AutoConfig

from miles.backends.megatron_utils.lora.slots import create_multi_lora_instance
from miles.backends.megatron_utils.lora.target_modules import resolve_megatron_lora_targets
from miles.backends.megatron_utils.lora.utils import create_lora_instance
from miles.utils.hf_utils.weight_mapping import HfWeightMapping


@pytest.fixture(params=["qkv", "experts", "gdn"])
def target_case(request):
    case = request.param
    config = AutoConfig.for_model(
        "qwen3_next" if case == "gdn" else "qwen3_moe",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        decoder_sparse_step=1,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    bridge = Qwen3NextBridge() if case == "gdn" else Qwen3MoEBridge()
    targets, module, parameters = {
        "qkv": (
            ["model.layers.*.self_attn.q_proj"],
            "decoder.layers.0.self_attention.linear_qkv",
            ["weight"],
        ),
        "experts": (
            ["model.layers.*.mlp.experts.gate_up_proj"],
            "decoder.layers.0.mlp.experts.linear_fc1",
            ["weight0", "weight1"],
        ),
        "gdn": (
            ["model.layers.*.linear_attn.in_proj_qkvz", "model.layers.*.linear_attn.in_proj_ba"],
            "decoder.layers.0.self_attention.in_proj",
            ["weight"],
        ),
    }[case]
    return (
        case,
        bridge.mapping_registry().get_all_mappings(),
        HfWeightMapping.from_config(config),
        targets,
        module,
        parameters,
    )


@pytest.mark.parametrize("mode", ["lora", "canonical_lora", "multi_lora"])
def test_registry_to_adapter_matcher(target_case, mode):
    case, mappings, hf_mapping, targets, module, parameters = target_case
    canonical = mode == "canonical_lora"

    def resolve(selection):
        # Explicit parameter fixtures exercise registry binding without constructing a distributed trainer.
        return resolve_megatron_lora_targets(
            selection,
            mappings,
            parameter_names={f"{module}.{parameter}" for parameter in parameters},
            hf_mapping=hf_mapping,
            canonical=canonical,
        )

    if case == "qkv" and not canonical:
        with pytest.raises(AssertionError, match="requires all HF targets"):
            resolve(targets)
        targets = [f"model.layers.*.self_attn.{projection}_proj" for projection in ("q", "k", "v")]
    if case == "gdn":
        with pytest.raises(AssertionError, match="requires all HF targets"):
            resolve(targets[:1])

    selected = resolve(targets)
    template = module.replace(".0.", ".*.")
    expected = {template}
    if canonical and case == "qkv":
        expected = {template.removesuffix("linear_qkv") + "linear_q"}
    elif canonical and case == "experts":
        expected = {template + "_gate", template + "_up"}
    assert set(selected) == expected

    args = SimpleNamespace(lora_type=mode, lora_rank=2, lora_alpha=4, lora_dropout=0.0, multi_lora_n_adapters=2)
    create_adapter = create_multi_lora_instance if mode == "multi_lora" else create_lora_instance
    adapter = create_adapter(args, target_modules=selected)
    adapter._init_target_match_state()
    linear = torch.nn.Linear(2, 2, device="meta")
    matched = set()
    for candidate in (module, "decoder.layers.0.mlp.router", "output_layer"):
        prefix, name = candidate.rsplit(".", 1) if "." in candidate else ("", candidate)
        if adapter.match(linear, name=name, prefix=prefix):
            matched.add(candidate)
    assert matched == {module}
    assert all(adapter._alias_matches[target] == {module} for target in selected)
