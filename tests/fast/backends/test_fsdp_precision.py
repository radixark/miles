"""Unit tests for the FSDP precision policy (CPU-only)."""

import sys
from types import SimpleNamespace

import torch

from miles.backends.experimental.fsdp_utils.adaptations.precision import apply_fp32_master, resolve_precision_policy
from miles.backends.experimental.fsdp_utils.arguments import parse_fsdp_cli


def test_resolve_precision_policy_uses_independent_fp32_master_switch_and_dtypes():
    dense = SimpleNamespace(model_type="qwen3")
    bf16_args = SimpleNamespace(fp16=False, enable_fp32_master=True)

    p = resolve_precision_policy(dense, bf16_args)
    assert p.keep_fp32_master
    assert p.param_dtype == torch.bfloat16 and p.reduce_dtype == torch.float32

    disabled = resolve_precision_policy(
        SimpleNamespace(model_type="glm4_moe_lite"),
        SimpleNamespace(fp16=True, enable_fp32_master=False),
    )
    assert not disabled.keep_fp32_master
    assert disabled.param_dtype == torch.float16 and disabled.reduce_dtype == torch.float32
    assert disabled == resolve_precision_policy(
        dense,
        SimpleNamespace(fp16=True, enable_fp32_master=False),
    )


def test_fp32_master_cli_defaults_enabled_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["miles"])
    assert parse_fsdp_cli().enable_fp32_master

    monkeypatch.setattr(sys, "argv", ["miles", "--no-enable-fp32-master"])
    assert not parse_fsdp_cli().enable_fp32_master


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
