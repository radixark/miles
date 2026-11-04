"""The trainer's frozen MXFP4 experts compute exactly what BF16 experts holding the dequantized weights
compute: a packed row tensor round-trips the checkpoint's bytes, and a TE grouped linear built on packed rows
matches a BF16 ``TEColumnParallelGroupedLinear`` forward and backward bit for bit, owning no BF16 copy and
taking no weight gradient."""

import os
import sys

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from miles.utils.mxfp4 import (  # noqa: E402
    MXFP4_GROUP_SIZE,
    dequantize_mxfp4,
    dequantize_mxfp4_rows,
    mxfp4_row_width,
    pack_mxfp4_rows,
    quantize_mxfp4,
    split_mxfp4_rows,
)
from miles_plugins.models.kimi_k3.mxfp4_experts import (  # noqa: E402
    MXFP4ColumnParallelGroupedLinear,
    MXFP4RowParallelGroupedLinear,
)

_LINEARS = {
    "fc1": (MXFP4ColumnParallelGroupedLinear, TEColumnParallelGroupedLinear),
    "fc2": (MXFP4RowParallelGroupedLinear, TERowParallelGroupedLinear),
}


@pytest.fixture(scope="module")
def config():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=3,
        moe_grouped_gemm=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
    )
    yield config
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _packed_experts(num: int, out_features: int, in_features: int):
    torch.manual_seed(1)
    encoded = [
        quantize_mxfp4(torch.randn(out_features, in_features, device="cuda"), MXFP4_GROUP_SIZE) for _ in range(num)
    ]
    rows = [pack_mxfp4_rows(packed, scale, MXFP4_GROUP_SIZE) for packed, scale in encoded]
    dequantized = [dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE) for packed, scale in encoded]
    return rows, dequantized


def test_packed_rows_round_trip(config):
    rows, dequantized = _packed_experts(2, 64, 128)
    for row, reference in zip(rows, dequantized, strict=True):
        assert row.dtype == torch.uint8 and row.shape == (64, mxfp4_row_width(128, MXFP4_GROUP_SIZE))
        packed, scale = split_mxfp4_rows(row, MXFP4_GROUP_SIZE)
        assert packed.shape == (64, 64) and scale.shape == (64, 4)
        assert torch.equal(dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE), reference)
        assert torch.equal(dequantize_mxfp4_rows(row, MXFP4_GROUP_SIZE), reference)


@pytest.mark.parametrize("linear", ["fc1", "fc2"])
def test_grouped_linear_matches_bf16_host(config, linear):
    mxfp4_cls, bf16_cls = _LINEARS[linear]
    num, in_features, out_features = 3, 256, 384
    kwargs = dict(config=config, init_method=torch.nn.init.zeros_, bias=False, skip_bias_add=False, is_expert=True)
    frozen = mxfp4_cls(num, in_features, out_features, **kwargs)
    plain = bf16_cls(num, in_features, out_features, **kwargs)
    rows, dequantized = _packed_experts(num, out_features, in_features)
    for i in range(num):
        frozen_weight = getattr(frozen, f"weight{i}")
        assert frozen_weight.dtype == torch.uint8 and not frozen_weight.requires_grad
        assert frozen_weight.shape == (out_features, mxfp4_row_width(in_features, MXFP4_GROUP_SIZE))
        with torch.no_grad():
            frozen_weight.copy_(rows[i])
            getattr(plain, f"weight{i}").copy_(dequantized[i])

    m_splits = torch.tensor([5, 3, 7])
    x = torch.randn(int(m_splits.sum()), in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_(True)
    out, _ = frozen(x, m_splits)
    reference, _ = plain(x_reference, m_splits)
    assert torch.equal(out, reference)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    reference.backward(grad_out)
    assert torch.equal(x.grad, x_reference.grad)
    assert all(getattr(frozen, f"weight{i}").grad is None for i in range(num))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
