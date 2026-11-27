from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.megatron_to_hf.qwen3_5 import convert_qwen3_5_to_hf
from miles_plugins.mbridge.linear_attn import LinearAttentionBridgeMixin

ARGS = SimpleNamespace(
    hf_checkpoint=None,
    true_on_policy_contract=None,
    backend=SimpleNamespace(kv_channels=None, hidden_size=64, num_attention_heads=4, num_query_groups=4),
)
PREFIX = "module.module.decoder.layers.3"


def test_sequential_mlp_expert_splits_gate_and_up():
    fc1 = convert_qwen3_5_to_hf(
        ARGS, f"{PREFIX}.mlp.experts.local_experts.5.linear_fc1.weight", torch.arange(8.0).view(4, 2)
    )
    assert [name for name, _ in fc1] == [
        "model.language_model.layers.3.mlp.experts.5.gate_proj.weight",
        "model.language_model.layers.3.mlp.experts.5.up_proj.weight",
    ]
    assert torch.equal(fc1[0][1], torch.arange(4.0).view(2, 2))
    fc2 = convert_qwen3_5_to_hf(ARGS, f"{PREFIX}.mlp.experts.local_experts.5.linear_fc2.weight", torch.ones(2, 2))
    assert [name for name, _ in fc2] == ["model.language_model.layers.3.mlp.experts.5.down_proj.weight"]


def test_standalone_input_layernorm():
    out = convert_qwen3_5_to_hf(ARGS, f"{PREFIX}.input_layernorm.weight", torch.ones(4))
    assert [name for name, _ in out] == ["model.language_model.layers.3.input_layernorm.weight"]


def test_linear_attention_mixin_passes_consider_ep():
    class Base:
        def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep=True):
            return {"a": f"x.{consider_ep}", "b": "y.weight_layout_version"}

    class Bridge(LinearAttentionBridgeMixin, Base):
        pass

    assert Bridge()._weight_name_mapping_mcore_local_to_global(None, consider_ep=False) == {"a": "x.False"}
