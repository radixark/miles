"""Merged export: every adapter's weight delta must reproduce its forward delta — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.merge import merge_lora_into_weights
from miles_plugins.lora.modules.linear import attach_adapter_forward
from miles_plugins.lora.modules.moe import LoRAGroupedFC1, LoRAGroupedFC2, LoRAOutputHead, _grouped_linear
from miles_plugins.lora.spec.attention import GQAAttentionSpec, MLAAttentionSpec
from miles_plugins.lora.spec.base import AttachContext
from miles_plugins.lora.spec.mlp import FusedGatedMLPSpec, InklingDenseMLPSpec
from miles_plugins.lora.spec.moe import InklingExpertsSpec

HIDDEN = 8


class _Linear(nn.Module):
    def __init__(self, out_features, in_features, *, fused_norm=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if fused_norm:
            self.layer_norm_weight = nn.Parameter(torch.rand(in_features) + 0.5)

    def forward(self, x):
        if hasattr(self, "layer_norm_weight"):
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.layer_norm_weight
        return x @ self.weight.t(), None


class _GroupedLinear(nn.Module):
    def __init__(self, num_experts, out_features, in_features):
        super().__init__()
        for index in range(num_experts):
            self.register_parameter(f"weight{index}", nn.Parameter(torch.randn(out_features, in_features)))

    def forward(self, x, tokens_per_expert):
        weights = torch.stack([getattr(self, f"weight{i}") for i in range(len(tokens_per_expert))])
        return _grouped_linear(x, weights, tokens_per_expert), None


def _context(targets=None, *, output_gate=False, **config):
    transformer_config = SimpleNamespace(
        hidden_size=HIDDEN,
        sequence_parallel=False,
        layernorm_epsilon=1e-5,
        attention_output_gate=output_gate,
        **config,
    )
    return AttachContext(
        lora=LoRAConfig(rank=2, alpha=6, dropout=0.0, target_modules=targets),
        transformer_config=transformer_config,
        tp_size=1,
        tp_rank=0,
        layer_prefix="model.layers.",
        shared_expert="mlp.shared_experts.",
    )


def _randomize_adapters(module):
    torch.manual_seed(1)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "lora" in name or name.split(".")[-1] in ("head_A", "head_B"):
                parameter.normal_()


def _merged(module):
    weights = {name: parameter for name, parameter in module.named_parameters() if "lora" not in name}
    return merge_lora_into_weights([module], weights)


def _assert_merge_matches_forward(module, names_and_hosts, *host_args):
    _randomize_adapters(module)
    merged = _merged(module)
    for name, host in names_and_hosts:
        x = torch.randn(5, host.weight.shape[1] if not isinstance(host, _GroupedLinear) else host.weight0.shape[1])
        expected = host(x, *host_args)[0]
        base_forward = host.forward
        del host.forward
        with torch.no_grad():
            for weight_name in [n for n, _ in host.named_parameters() if n.startswith("weight")]:
                getattr(host, weight_name).copy_(merged[f"{name}.{weight_name}"])
            actual = host(x, *host_args)[0]
        host.forward = base_forward
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4, msg=name)


def _gqa_attention(heads=4, groups=2, head_dim=2, output_gate=False):
    attention = nn.Module()
    q_rows = heads * head_dim * (2 if output_gate else 1)
    attention.linear_qkv = _Linear(q_rows + 2 * groups * head_dim, HIDDEN, fused_norm=True)
    attention.linear_proj = _Linear(HIDDEN, heads * head_dim)
    attention.num_attention_heads_per_partition = heads
    attention.num_query_groups_per_partition = groups
    attention.hidden_size_per_attention_head = head_dim
    return attention


@pytest.mark.parametrize("output_gate", [False, True])
@pytest.mark.parametrize("targets", [None, ("q_proj", "v_proj")], ids=["qkvo", "partial-qv"])
def test_gqa_fused_qkv_and_row_parallel_o(output_gate, targets):
    module = nn.Module()
    module.attn = _gqa_attention(output_gate=output_gate)
    GQAAttentionSpec().attach(module.attn, "model.layers.0.self_attn.", _context(targets, output_gate=output_gate))
    hosts = [("attn.linear_qkv", module.attn.linear_qkv)]
    if targets is None:
        hosts.append(("attn.linear_proj", module.attn.linear_proj))
    _assert_merge_matches_forward(module, hosts)


def test_mla_replicated_and_column_projections():
    config = dict(q_lora_rank=4, kv_lora_rank=3, qk_pos_emb_head_dim=1, qk_head_dim=2, v_head_dim=2)
    attention = nn.Module()
    attention.num_attention_heads_per_partition, attention.q_head_dim = 2, 3
    attention.linear_q_down_proj = _Linear(4, HIDDEN)
    attention.linear_q_up_proj = _Linear(6, 4)
    attention.linear_kv_down_proj = _Linear(4, HIDDEN)
    attention.linear_kv_up_proj = _Linear(8, 3)
    attention.linear_proj = _Linear(HIDDEN, 4)
    module = nn.Module()
    module.attn = attention
    MLAAttentionSpec().attach(attention, "model.layers.0.self_attn.", _context(**config))
    _assert_merge_matches_forward(
        module, [(f"attn.{name}", getattr(attention, name)) for name in ("linear_q_down_proj", "linear_kv_up_proj")]
    )


@pytest.mark.parametrize("spec", [FusedGatedMLPSpec(), InklingDenseMLPSpec()], ids=["split-gate-up", "tml-fused"])
def test_gated_mlp(spec):
    mlp = nn.Module()
    mlp.linear_fc1, mlp.linear_fc2 = _Linear(10, HIDDEN, fused_norm=True), _Linear(HIDDEN, 5)
    module = nn.Module()
    module.mlp = mlp
    spec.attach(mlp, "model.layers.0.mlp.", _context())
    _assert_merge_matches_forward(module, [("mlp.linear_fc1", mlp.linear_fc1), ("mlp.linear_fc2", mlp.linear_fc2)])


def test_grouped_routed_experts():
    context = _context()
    experts = nn.Module()
    experts.linear_fc1, experts.linear_fc2 = _GroupedLinear(3, 10, HIDDEN), _GroupedLinear(3, HIDDEN, 5)
    common = dict(reference=experts.linear_fc1.weight0, context=context, num_local_experts=3, moe_intermediate=5)
    experts.lora_fc1_adapter = LoRAGroupedFC1(hf_prefix="e.", is_ep=False, **common)
    experts.lora_fc2_adapter = LoRAGroupedFC2(hf_prefix="e.", is_ep=False, **common)
    attach_adapter_forward(experts.linear_fc1, experts.lora_fc1_adapter, context.scale)
    attach_adapter_forward(experts.linear_fc2, experts.lora_fc2_adapter, context.scale)
    module = nn.Module()
    module.experts = experts
    _assert_merge_matches_forward(
        module, [("experts.linear_fc1", experts.linear_fc1), ("experts.linear_fc2", experts.linear_fc2)], [2, 2, 1]
    )


def test_inkling_shared_sub_experts():
    shared = nn.Module()
    shared.experts = nn.ModuleList()
    for _ in range(2):
        sub = nn.Module()
        sub.linear_fc1, sub.linear_fc2 = _Linear(10, HIDDEN), _Linear(HIDDEN, 5)
        shared.experts.append(sub)
    InklingExpertsSpec._attach_shared(shared, "model.layers.1.", _context())
    module = nn.Module()
    module.shared = shared
    _assert_merge_matches_forward(
        module,
        [
            (f"shared.experts.{index}.{name}", getattr(shared.experts[index], name))
            for index in range(2)
            for name in ("linear_fc1", "linear_fc2")
        ],
    )


def test_output_head_with_mup_input_scaling():
    context = _context()
    module = nn.Module()
    module.output_layer = _Linear(12, HIDDEN)
    module.head = LoRAOutputHead(
        hf_prefix="lm_head.",
        reference=module.output_layer.weight,
        context=context,
        vocab_local=12,
        mup_width_multiplier=4.0,
    )
    attach_adapter_forward(module.output_layer, module.head, context.scale)
    _assert_merge_matches_forward(module, [("output_layer", module.output_layer)])


def test_merge_refuses_a_host_weight_that_is_not_exported():
    module = nn.Module()
    module.mlp = nn.Module()
    module.mlp.linear_fc1, module.mlp.linear_fc2 = _Linear(10, HIDDEN), _Linear(HIDDEN, 5)
    FusedGatedMLPSpec().attach(module.mlp, "model.layers.0.mlp.", _context())
    with pytest.raises(AssertionError, match="not exported"):
        merge_lora_into_weights([module], {"mlp.linear_fc1.weight": module.mlp.linear_fc1.weight})


def test_unadapted_weights_pass_through_untouched():
    module = nn.Module()
    module.mlp = nn.Module()
    module.mlp.linear_fc1, module.mlp.linear_fc2 = _Linear(10, HIDDEN), _Linear(HIDDEN, 5)
    module.norm = nn.Parameter(torch.ones(HIDDEN))
    FusedGatedMLPSpec().attach(module.mlp, "model.layers.0.mlp.", _context(("down_proj",)))
    merged = _merged(module)
    assert merged["norm"] is module.norm
    assert merged["mlp.linear_fc1.weight"] is module.mlp.linear_fc1.weight
