"""Unit tests for the native-LoRA adapter modules and forward hooks — no GPU."""

from types import SimpleNamespace

import torch

from miles_plugins.lora.distributed import rmsnorm
from miles_plugins.lora.lora import _require_grad_on_first_activation
from miles_plugins.lora.modules.linear import build_qkv_permutation


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


class TestRmsNorm:
    def test_plain_gamma_scales_by_the_stored_weight(self):
        x = torch.tensor([[3.0, 4.0]])
        gamma = torch.tensor([2.0, 2.0])
        got = rmsnorm(x, gamma, eps=0.0)
        assert torch.allclose(got, torch.tensor([[3.0, 4.0]]) / 3.5355339 * 2.0, atol=1e-5)

    def test_zero_centered_gamma_adds_the_one_back(self):
        """--apply-layernorm-1p stores gamma - 1; the branch must see the same input
        the base GEMM does, or the adapter is fed a differently scaled activation."""
        x = torch.tensor([[3.0, 4.0]])
        stored = torch.tensor([1.0, 1.0])
        assert torch.allclose(
            rmsnorm(x, stored, eps=0.0, zero_centered_gamma=True),
            rmsnorm(x, stored + 1.0, eps=0.0),
        )


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

    def test_stage_without_an_embedding_is_a_noop(self):
        assert _require_grad_on_first_activation(SimpleNamespace()) is None
