"""GLM-5.3-Flash VJP reference: checkpoint names as the reference layer's, the per-token mHC forward, selection refusals."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

from miles.utils.debug_utils.top_reference_glm53 import (
    key_indices,
    require_selection,
    to_reference_names,
    token_terms,
    top_name,
)


def test_checkpoint_names_become_the_reference_layers():
    """mHC and forget-gate parameters are renamed, the q/k/v convolutions join in that order, and the frozen indexer and
    the router's bias buffer have no reference gradient."""
    q, k, v = (torch.full((2, 1, 4), float(i)) for i in range(3))
    out = to_reference_names(
        {
            "model.layers.0.hc_attn_fn": torch.zeros(24, 8),
            "model.layers.0.hc_ffn_scale": torch.zeros(3),
            "model.layers.0.self_attn.A_log": torch.zeros(4),
            "model.layers.0.self_attn.f_b_proj.weight": torch.zeros(8, 2),
            "model.layers.0.self_attn.q_conv1d.weight": q,
            "model.layers.0.self_attn.k_conv1d.weight": k,
            "model.layers.0.self_attn.v_conv1d.weight": v,
            "model.layers.1.self_attn.indexer.wk.weight": torch.zeros(1),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.zeros(1),
            "model.layers.1.mlp.gate.weight": torch.zeros(1),
        }
    )
    assert set(out) == {
        "model.layers.0.attn_hc.fn",
        "model.layers.0.ffn_hc.scale",
        "model.layers.0.self_attn.forget_gate.A_log",
        "model.layers.0.self_attn.forget_gate.f_b_proj.weight",
        "model.layers.0.self_attn.conv1d.weight",
        "model.layers.1.mlp.gate.weight",
    }
    assert torch.equal(out["model.layers.0.self_attn.conv1d.weight"], torch.cat([q, k, v]))


def _layer():
    config = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5, hidden_size=8)
    torch.manual_seed(0)
    layer = torch.nn.Module()
    for site in ("attn_hc", "ffn_hc"):
        hc = Glm5NextTextHyperConnection(config)
        for param in hc.parameters():
            torch.nn.init.normal_(param, std=0.5)
        layer.add_module(site, hc)
    return layer


def test_the_per_token_mhc_forward_is_the_references_and_records_token_sums():
    """Under token_terms the base and scale are taken per token: at the FP32 replay's dtype the outputs are the
    reference's bit for bit, the parameter gradients agree, and sum_t |g_t| is recorded for each."""
    layer = _layer()
    streams = torch.randn(1, 5, 4, 8)
    reference = layer.attn_hc(streams)
    sum(out.sum() for out in reference).backward()
    expected = {name: p.grad.clone() for name, p in layer.attn_hc.named_parameters()}
    layer.zero_grad()
    with token_terms(layer) as sums:
        outputs = layer.attn_hc(streams)
        sum(out.sum() for out in outputs).backward()
    for ours, theirs in zip(outputs, reference, strict=True):
        assert torch.equal(ours, theirs)
    for name, param in layer.attn_hc.named_parameters():
        torch.testing.assert_close(param.grad, expected[name], rtol=1e-5, atol=1e-7)
    assert set(sums) == {"attn_hc.base", "attn_hc.scale"}
    assert all(total > 0 for total in sums.values())
    assert "forward" not in vars(layer.attn_hc)


def test_a_dsa_layer_needs_a_captured_selection_and_a_kda_layer_none():
    dsa, kda = SimpleNamespace(block_type="deepseek_sparse_attention"), SimpleNamespace(block_type="linear_attention")
    with pytest.raises(ValueError, match="captured index selection"):
        require_selection(dsa, None)
    with pytest.raises(ValueError, match="KDA layer has none"):
        require_selection(kda, torch.zeros(1))
    require_selection(dsa, torch.zeros(1))
    require_selection(kda, None)


def test_only_mhc_bases_have_a_top_name():
    assert top_name("model.layers.3.ffn_hc.base") == "decoder.layers.3.mlp_hyper_connection.bias"
    assert top_name("model.layers.3.attn_hc.base") == "decoder.layers.3.self_attention_hyper_connection.bias"
    assert top_name("model.layers.3.attn_hc.scale") is None


def test_a_row_that_skips_its_own_key_keeps_it_unselected():
    """Rows 10..13 of the packed stream are one document. Row 3 closes a pool the indexer did not pick, so it selects
    neither that pool nor itself; its -1 pads must stay no key, not stand in for its own position."""
    indices = torch.tensor([[10, -1, -1], [10, 11, -1], [10, 11, 12], [10, -1, -1]], dtype=torch.int32)[:, None]
    keys = key_indices(indices, lo=10)
    assert torch.equal(keys[0], torch.tensor([[0, -1, -1], [0, 1, -1], [0, 1, 2], [0, -1, -1]]))


@pytest.mark.parametrize(
    ("indices", "refusal"),
    [
        ([[10, 11], [10, 11]], "outside its row's causal keys"),  # row 0 selects key 1, after itself
        ([[9, -1], [10, 11]], "outside its row's causal keys"),
        ([[-1, -1], [10, 11]], "no key"),
    ],
)
def test_a_selection_the_reference_cannot_replay_is_refused(indices, refusal):
    with pytest.raises(ValueError, match=refusal):
        key_indices(torch.tensor(indices, dtype=torch.int32)[:, None], lo=10)
