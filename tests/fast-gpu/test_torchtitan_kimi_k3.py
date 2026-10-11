"""K3 checkpoint conversion, packed boundaries, and expert reduction precision."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci
from torchtitan.models.common.token_dispatcher import LocalDispatchMetadata

from miles.backends.torchtitan_utils.models.kimi_k3.attention import MLAAttention
from miles.backends.torchtitan_utils.models.kimi_k3.model import KimiK3Model
from miles.backends.torchtitan_utils.models.kimi_k3.ops import KimiTokenDispatcher, packed_metadata
from miles.backends.torchtitan_utils.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-b200", num_gpus=1, labels=["torchtitan"], hardware=["blackwell"])


def test_packed_documents_and_rows_do_not_attend_to_each_other():
    tokens = torch.zeros(2, 4, dtype=torch.long)
    positions = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]])
    cumulative, mask = packed_metadata(tokens, positions)
    assert cumulative.tolist() == [0, 2, 4, 8]
    for query in range(8):
        start = 0 if query < 2 else 2 if query < 4 else 4
        assert mask[query].nonzero().flatten().tolist() == list(range(start, query + 1))


def test_explicit_boundaries_keep_padding_together_and_single_token_documents():
    tokens = torch.zeros(1, 8, dtype=torch.long)
    positions = torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0]])
    boundaries = torch.tensor([0, 3, 4, 8], dtype=torch.int32)
    cumulative, mask = packed_metadata(tokens, positions, boundaries)
    assert cumulative.tolist() == [0, 3, 4, 8]
    assert mask[3].nonzero().flatten().tolist() == [3]
    assert mask[7].nonzero().flatten().tolist() == [4, 5, 6, 7]
    _, inferred_mask = packed_metadata(tokens, positions)
    torch.testing.assert_close(mask[:4], inferred_mask[:4])


def test_hf_roundtrip_preserves_padded_kda_and_grouped_experts():
    adapter = KimiK3StateDictAdapter(KimiK3Model.Config(num_experts=2), None)
    state = {
        "layers.0.self_attn.core.A_log": torch.arange(128, dtype=torch.float32),
        "layers.0.self_attn.core.dt_bias": torch.arange(8, dtype=torch.float32),
        "layers.1.moe.gate.e_score_correction_bias": torch.tensor([0.25, -0.25]),
        "tok_embeddings.weight": torch.randn(8, 4),
        "lm_head.weight": torch.randn(8, 4),
    }
    for name, shape in (("w1_EFD", (2, 3, 4)), ("w2_EDF", (2, 4, 3)), ("w3_EFD", (2, 3, 4))):
        state["layers.1.moe.routed_experts.inner_experts." + name] = torch.randn(shape)
    hf = adapter.to_hf(state)
    assert hf["language_model.model.layers.0.self_attn.A_log"].shape == (128,)
    actual = adapter.from_hf(hf)
    assert actual.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(actual[name], state[name], rtol=0, atol=0)
    hf.pop("language_model.model.layers.1.block_sparse_moe.experts.1.w2.weight")
    with pytest.raises(ValueError, match="Incomplete K3 expert groups"):
        adapter.from_hf(hf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scatter reduction")
def test_expert_sum_rounds_once_and_preserves_router_and_expert_gradients():
    dispatcher = KimiTokenDispatcher.Config(num_experts=3, top_k=3).build()
    values = torch.tensor([[512, 1], [1, 1], [-512, 1]], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores = torch.ones(3, device="cuda", requires_grad=True)
    metadata = LocalDispatchMetadata(
        token_indices_experts_sorted_N=torch.zeros(3, device="cuda", dtype=torch.long),
        topk_scores_experts_sorted_N=scores,
    )
    output = dispatcher.combine(values, metadata, values.new_zeros(1, 2))
    torch.testing.assert_close(
        output.float(), torch.tensor([[1, 3]], device="cuda"), rtol=0, atol=0, check_dtype=False
    )
    output.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)
    torch.testing.assert_close(scores.grad, torch.tensor([513.0, 2.0, -511.0], device="cuda"), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA and grouped expert kernels require CUDA")
def test_small_decoder_backward_and_adam_update():
    torch.manual_seed(42)
    config = KimiK3Model.Config(
        dim=128,
        vocab_size=256,
        n_layers=4,
        n_heads=4,
        mla_heads=4,
        num_experts=4,
        top_k=2,
        latent_dim=64,
        dense_intermediate=256,
        moe_intermediate=128,
        q_lora_rank=64,
        kv_lora_rank=64,
    )
    model = config.build()
    model.verify_module_protocol()
    model.init_states()
    model.cuda()
    tokens = torch.arange(32, device="cuda").reshape(1, 32)
    boundaries = torch.tensor([0, 15, 16, 32], dtype=torch.int32, device="cuda")
    before = model.layers["0"].self_attn.q_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(tokens, cu_seqlens=boundaries)
    assert logits.shape == (1, 32, 256)
    torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), tokens.flatten()).backward()
    for name, param in model.named_parameters():
        # Layer zero's pre-attention residual projection is unused while the bank is empty.
        if name.startswith("layers.0.self_attention_res_"):
            continue
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    optimizer.step()
    assert not torch.equal(before, model.layers["0"].self_attn.q_proj.weight)


def test_mla_bottleneck_norms_follow_checkpoint_epsilon():
    config = KimiK3Model.Config(dim=16, mla_heads=1, q_lora_rank=8, kv_lora_rank=4, norm_eps=1e-5)
    attention = MLAAttention(config)
    for norm, width in ((attention.q_a_layernorm, 8), (attention.kv_a_layernorm, 4)):
        values = torch.full((1, width), 0.001)
        expected = values / (0.001**2 + config.norm_eps) ** 0.5
        torch.testing.assert_close(norm(values), expected)
