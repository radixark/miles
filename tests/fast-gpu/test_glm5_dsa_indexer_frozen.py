"""The Miles DSA indexer only selects top-k indices, which carry no gradient, and has no
auxiliary loss, so its parameters must be frozen: otherwise DDP and the optimizer keep them
and weight decay shrinks them every step, drifting the top-k the rollout engine serves."""

import os

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=50,
    suite="stage-b-2-gpu-h200",
    labels=["miles-plugin"],
    hardware=["hopper", "blackwell"],
    num_gpus=1,
    nightly=True,
)

pytest.importorskip("transformer_engine")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType  # noqa: E402
from megatron.core.transformer.identity_op import IdentityOp  # noqa: E402
from megatron.core.transformer.spec_utils import ModuleSpec, build_module  # noqa: E402
from megatron.core.transformer.transformer_config import MLATransformerConfig  # noqa: E402

from miles_plugins.models.glm5.glm5 import DSAMLASelfAttention, DSASelfAttentionSubmodules  # noqa: E402


@pytest.fixture
def model_parallel(tmp_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _attention() -> DSAMLASelfAttention:
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        bf16=True,
        multi_latent_attention=True,
    )
    config.index_num_attention_heads = 2
    config.index_head_dim = 32
    config.indexer_rope_interleave = False
    config.index_topk_freq = 1
    config.index_skip_topk_offset = 0
    backend = TESpecProvider()
    spec = ModuleSpec(
        module=DSAMLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal, "topk_backend": "torch"},
        submodules=DSASelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_layer_norm_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
            linear_v_up_proj=IdentityOp,
            wq_b=backend.linear(),
            wk=backend.linear(),
            k_norm=backend.layer_norm(),
            weights_proj=backend.linear(),
        ),
    )
    return build_module(spec, config=config, layer_number=1)


def test_indexer_is_frozen_and_attention_trains(model_parallel) -> None:
    attention = _attention()
    trainable = {name for name, param in attention.named_parameters() if param.requires_grad}
    frozen = {name for name, param in attention.named_parameters() if not param.requires_grad}

    # Spelled out rather than read from _INDEXER_SUBMODULE_NAMES, so a module missing there fails too.
    assert frozen == {"wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight"}
    assert {"linear_q_down_proj.weight", "linear_kv_down_proj.weight", "linear_proj.weight"} <= trainable


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
