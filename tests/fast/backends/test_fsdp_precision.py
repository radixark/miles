"""Unit tests for the FSDP precision policy (CPU-only)."""

from types import SimpleNamespace

import torch

from miles.backends.experimental.fsdp_utils.adaptations.precision import apply_fp32_master, resolve_precision_policy


def test_resolve_precision_policy_gating_and_dtypes():
    dense = SimpleNamespace(model_type="qwen3")
    bf16_args = SimpleNamespace(fp16=False)

    # fp32 master only for glm4_moe_lite; reduce is always fp32; param follows args.fp16
    assert resolve_precision_policy(SimpleNamespace(model_type="glm4_moe_lite"), bf16_args).keep_fp32_master
    p = resolve_precision_policy(dense, bf16_args)
    assert not p.keep_fp32_master
    assert p.param_dtype == torch.bfloat16 and p.reduce_dtype == torch.float32
    assert resolve_precision_policy(dense, SimpleNamespace(fp16=True)).param_dtype == torch.float16
    # nemotron / qwen3_moe must NOT keep an fp32 master
    assert not resolve_precision_policy(SimpleNamespace(model_type="nemotron_h"), bf16_args).keep_fp32_master
    assert not resolve_precision_policy(SimpleNamespace(model_type="qwen3_moe"), bf16_args).keep_fp32_master


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
