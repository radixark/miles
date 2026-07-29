import torch

from miles_plugins.models.deepseek_v4.ops.utils import dsv4_query_rms_norm


def test_dsv4_query_rms_is_batch_invariant():
    torch.manual_seed(1234)
    full = torch.randn(3, 17, 4, 512, dtype=torch.bfloat16)
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
