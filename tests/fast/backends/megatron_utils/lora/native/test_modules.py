"""Unit tests for the native-LoRA adapter modules and forward hooks — no GPU."""

import argparse
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.lora import _require_grad_on_first_activation
from miles.backends.megatron_utils.lora.native.modules.linear import (
    LoRALinear,
    attach_adapter_forward,
    build_qkv_permutation,
    new_lora_parameter,
)
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, ProjectionSpec, ShardLayout
from miles.utils.lora.arguments import add_lora_arguments


class TestBuildQkvPerm:
    def test_mha_single_group(self):
        perm = build_qkv_permutation(num_q_heads=1, num_groups=1, head_dim=2, device="cpu")
        assert perm.tolist() == [0, 1, 2, 3, 4, 5]

    def test_gqa_two_groups_matches_mcore_layout(self):
        perm = build_qkv_permutation(num_q_heads=4, num_groups=2, head_dim=1, device="cpu")
        assert perm.tolist() == [0, 1, 4, 6, 2, 3, 5, 7]

    def test_permutation_is_a_bijection(self):
        nq, ng, hd = 8, 4, 3
        perm = build_qkv_permutation(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu")
        total = (nq + 2 * ng) * hd
        assert perm.numel() == total
        assert sorted(perm.tolist()) == list(range(total))

    def test_applied_to_delta_places_projections_per_group(self):
        nq, ng, hd = 4, 2, 1
        perm = build_qkv_permutation(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu")
        plain = torch.tensor([[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 30.0, 31.0]])
        out = plain.index_select(-1, perm)
        assert out.tolist() == [[10.0, 11.0, 20.0, 30.0, 12.0, 13.0, 21.0, 31.0]]

    def test_output_gate_deinterleaves_the_query_slices(self):
        perm = build_qkv_permutation(num_q_heads=2, num_groups=1, head_dim=1, device="cpu", output_gate=True)
        assert perm.tolist() == [0, 2, 1, 3, 4, 5]

    def test_output_gate_permutation_is_a_bijection(self):
        nq, ng, hd = 8, 2, 3
        perm = build_qkv_permutation(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu", output_gate=True)
        total = (2 * nq + 2 * ng) * hd
        assert perm.numel() == total
        assert sorted(perm.tolist()) == list(range(total))

    def test_output_gate_applied_to_delta(self):
        perm = build_qkv_permutation(num_q_heads=4, num_groups=2, head_dim=1, device="cpu", output_gate=True)
        plain = torch.tensor([[10.0, 40.0, 11.0, 41.0, 12.0, 42.0, 13.0, 43.0, 20.0, 21.0, 30.0, 31.0]])
        out = plain.index_select(-1, perm)
        assert out.tolist() == [[10.0, 11.0, 40.0, 41.0, 20.0, 30.0, 12.0, 13.0, 42.0, 43.0, 21.0, 31.0]]


class TestFirstActivationGrad:
    """A frozen base plus recomputation is the case that silently trains nothing.

    Every adapter param sits inside a checkpointed block, so unless the block's
    input requires grad, autograd never enters the region and every adapter
    gradient comes back zero while all the sync checks still pass.
    """

    def test_a_frozen_embedding_output_has_no_graph_on_its_own(self):
        embedding = torch.nn.Embedding(4, 3)
        embedding.weight.requires_grad_(False)
        assert not embedding(torch.tensor([0, 1])).requires_grad

    def test_hook_makes_the_first_activation_require_grad(self):
        embedding = torch.nn.Embedding(4, 3)
        embedding.weight.requires_grad_(False)
        model = SimpleNamespace(embedding=embedding)
        assert _require_grad_on_first_activation(model) is embedding
        assert embedding(torch.tensor([0, 1])).requires_grad

    def test_eval_forward_stays_out_of_autograd(self):
        embedding = torch.nn.Embedding(4, 3)
        embedding.weight.requires_grad_(False)
        _require_grad_on_first_activation(SimpleNamespace(embedding=embedding))
        embedding.eval()
        assert not embedding(torch.tensor([0, 1])).requires_grad

    def test_stage_without_an_embedding_is_a_noop(self):
        assert _require_grad_on_first_activation(SimpleNamespace()) is None


class TestLoRAInitMethod:
    def test_cli_accepts_only_implemented_methods(self, capsys):
        parser = argparse.ArgumentParser()
        add_lora_arguments(parser)
        assert parser.parse_args([]).lora_A_init_method == "xavier"
        assert parser.parse_args(["--lora-A-init-method", "kaiming"]).lora_A_init_method == "kaiming"
        with pytest.raises(SystemExit):
            parser.parse_args(["--lora-A-init-method", "orthogonal"])
        assert "invalid choice" in capsys.readouterr().err

    def test_each_method_draws_its_own_distribution(self):
        """A silently substituted method would leave kaiming/normal at xavier's scale."""
        reference = torch.empty(0)
        rank, fan_in = 64, 4096

        def draw(init):
            torch.manual_seed(0)
            return new_lora_parameter(reference, (rank, fan_in), init=init)

        xavier, kaiming, normal = draw("xavier"), draw("kaiming"), draw("normal")
        assert xavier.abs().max() <= math.sqrt(6 / (rank + fan_in))
        assert kaiming.abs().max() <= 1 / math.sqrt(fan_in)
        assert kaiming.abs().max() > 0.9 / math.sqrt(fan_in)
        torch.testing.assert_close(normal.std().item(), 0.2, atol=5e-3, rtol=0)
        with pytest.raises(KeyError):
            draw("orthogonal")


class _TELayerNormLinear(nn.Module):
    """TE LayerNormLinear's contract: a LayerNorm with bias here, so a recomputed RMSNorm cannot match."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.layer_norm_weight = nn.Parameter(torch.rand(in_features) + 0.5)
        self.layer_norm_bias = nn.Parameter(torch.randn(in_features))
        self.return_layernorm_output = False
        self.return_layernorm_output_gathered = False
        self.te_return_bias = False

    def normed(self, x):
        return F.layer_norm(x, x.shape[-1:], self.layer_norm_weight, self.layer_norm_bias)

    def forward(self, x):
        normed = self.normed(x)
        out = normed @ self.weight.t()
        return ((out, normed) if self.return_layernorm_output else out), None


def test_a_column_adapter_reads_its_fused_norm_hosts_own_norm_output():
    """The adapter must see exactly the activation the base GEMM sees, whatever the norm is."""
    host = _TELayerNormLinear(4, 6)
    context = AttachContext(
        lora=LoRAConfig(rank=2, alpha=2, dropout=0.0, target_modules=None),
        transformer_config=SimpleNamespace(hidden_size=6, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )
    adapter = LoRALinear(
        hf_prefix="model.layers.0.mlp.",
        projection=ProjectionSpec("up_proj", "up", ShardLayout.COLUMN),
        reference=host.weight,
        context=context,
        in_features=6,
        out_features=4,
    )
    with torch.no_grad():
        adapter.up_A.normal_()
        adapter.up_B.normal_()
    attach_adapter_forward(host, adapter, context.lora.scale)
    x = torch.randn(3, 6)

    out, bias = host(x)

    assert bias is None
    expected = host.normed(x) @ (host.weight + context.lora.scale * adapter.up_B @ adapter.up_A).t()
    torch.testing.assert_close(out, expected)
    assert host.return_layernorm_output and not host.return_layernorm_output_gathered
