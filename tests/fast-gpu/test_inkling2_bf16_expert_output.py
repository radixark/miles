"""Inkling2's routed experts return their bf16 fc2 rows and the top-k combine sums them in fp32
(moe_permute_accumulation_in_fp32). Upcasting each routed row to fp32 inside the expert gave the
same sum but kept an extra [rows, hidden] fp32 tensor on the rank."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("transformer_engine")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from types import SimpleNamespace  # noqa: E402

from megatron.core.transformer.moe.moe_utils import unpermute  # noqa: E402

from miles_plugins.models.inkling2.model import build_inkling2_config  # noqa: E402

_TEXT_CONFIG = dict(
    hidden_size=256, num_hidden_layers=4, vocab_size=1024, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, d_rel=16, rel_extent=64, log_scaling_n_floor=128000, log_scaling_alpha=0.1, rms_norm_eps=1e-6,
    pre_norm_pre_affine=False, final_norm_pre_affine=False, use_value_norm_no_gamma=True,
    qk_l2_clip_threshold=22.6, rel_l2_clip_threshold=8.0, local_layer_ids=[0, 1, 3], sconv_kernel_size=4,
    unpadded_vocab_size=1000, final_logit_softcapping=30.0, swa_head_dim=64, swa_num_attention_heads=4,
    swa_num_key_value_heads=2, sliding_window_size=32, n_layers_per_attn_res_block=2,
    attn_res_normalize_values=True, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=0,
    intermediate_size=128, mlp_gate_softcap=4.0, mlp_up_softcap=25.0, route_scale=8.0, use_gate_bias=True,
)  # fmt: skip
_BACKEND = SimpleNamespace(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1,
    expert_tensor_parallel_size=1, context_parallel_size=1, sequence_parallel=False, bf16=True,
    recompute_granularity=None, recompute_method=None, recompute_num_layers=None,
    decoder_first_pipeline_num_layers=None, decoder_last_pipeline_num_layers=None,
    virtual_pipeline_model_parallel_size=None, pipeline_model_parallel_layout=None,
    account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False,
)  # fmt: skip


def test_config_sums_the_combine_in_fp32() -> None:
    config = build_inkling2_config(_TEXT_CONFIG, _BACKEND)
    assert config.moe_permute_accumulation_in_fp32 and not config.moe_permute_fusion


@pytest.fixture
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def test_fp32_accumulation_of_bf16_rows_matches_fp32_rows(deterministic) -> None:
    # Large enough that summing top-k bf16 rows in bf16 rounds differently.
    tokens, topk, hidden = 4096, 8, 512
    torch.manual_seed(0)
    rows = torch.randn(tokens * topk, hidden, device="cuda").to(torch.bfloat16)
    sorted_indices = torch.randperm(tokens * topk, device="cuda") % tokens
    shape = torch.Size((tokens, hidden))

    before = unpermute(rows.float(), sorted_indices, shape)  # the old fp32 expert output
    after = unpermute(rows, sorted_indices, shape, accumulate_in_fp32=True)

    assert after.dtype == torch.bfloat16
    # The layer rounds the MoE output to bf16 before its short convolution, so both paths
    # round the same fp32 sum once.
    torch.testing.assert_close(after, before.to(torch.bfloat16), rtol=0, atol=0)
    bf16_sum = unpermute(rows, sorted_indices, shape)
    assert not torch.equal(bf16_sum, after), "the fp32 accumulation must change the bf16 sum"
