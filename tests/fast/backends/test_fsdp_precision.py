"""Unit tests for the FSDP precision policy (CPU-only)."""

import torch

from miles.backends.experimental.fsdp_utils.adaptations.precision import apply_fp32_master


def test_apply_fp32_master_records_on_disk_dtypes_before_cast():
    m = torch.nn.Linear(4, 4).to(torch.bfloat16)
    # an fp32-on-disk param (e.g. glm's e_score_correction_bias) must be recorded as fp32 so the
    # weight-sync downcast keeps it fp32 -- casting it to bf16 would flip MoE routing.
    m.register_parameter("score_bias", torch.nn.Parameter(torch.zeros(4, dtype=torch.float32)))

    m = apply_fp32_master(m)

    # the master is fully fp32...
    assert all(p.dtype == torch.float32 for p in m.parameters())
    # ...but the recorded on-disk dtypes are the pre-cast ones (bf16 weight/bias, fp32 score_bias)
    od = m._fsdp_sync_orig_dtypes
    assert od["weight"] == torch.bfloat16 and od["bias"] == torch.bfloat16
    assert od["score_bias"] == torch.float32
