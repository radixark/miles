import torch

from miles_plugins.models.deepseek_v4.ops.utils import dsv4_query_rms_norm


def test_dsv4_query_rms_is_batch_invariant():
    # This FP32 shape reproduces a native CUDA reduction-association mismatch
    # on GB200 with torch 2.11.0+cu130. Only require fixed-tree equality here:
    # native-path inequality depends on the device and selected kernel.
    torch.manual_seed(0)
    full = torch.randn(3, 2, 4, 512, dtype=torch.float32)
    single = full[:, :1].clone()

    single_output = dsv4_query_rms_norm(
        single,
        eps=1.0e-6,
        batch_invariant=True,
    )
    full_output = dsv4_query_rms_norm(
        full,
        eps=1.0e-6,
        batch_invariant=True,
    )[:, :1]

    assert torch.equal(single_output, full_output)
