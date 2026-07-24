"""Unit tests for the FSDP precision policy (CPU-only)."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from miles.backends.experimental.fsdp_utils.adaptations.precision import (
    apply_fp32_master,
    precision_forward_context,
    resolve_precision_policy,
)
from miles.backends.experimental.fsdp_utils.arguments import load_fsdp_args, parse_fsdp_cli
from miles.backends.training_utils.data import _rollout_logprob_dtype
from miles.true_on_policy.contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1


def test_resolve_precision_policy_uses_independent_fp32_master_switch_and_dtypes():
    dense = SimpleNamespace(model_type="qwen3")
    bf16_args = SimpleNamespace(fp16=False, keep_fp32_master=True)

    p = resolve_precision_policy(dense, bf16_args)
    assert p.keep_fp32_master
    assert p.param_dtype == torch.bfloat16 and p.reduce_dtype == torch.float32

    disabled = resolve_precision_policy(
        SimpleNamespace(model_type="glm4_moe_lite"),
        SimpleNamespace(fp16=True, keep_fp32_master=False),
    )
    assert not disabled.keep_fp32_master
    assert disabled.param_dtype == torch.float16 and disabled.reduce_dtype == torch.float32
    assert disabled == resolve_precision_policy(
        dense,
        SimpleNamespace(fp16=True, keep_fp32_master=False),
    )


def test_fp32_master_cli_defaults_enabled_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["miles"])
    assert parse_fsdp_cli().keep_fp32_master

    monkeypatch.setattr(sys, "argv", ["miles", "--disable-fp32-master"])
    assert not parse_fsdp_cli().keep_fp32_master


def test_fsdp_args_expose_effective_compute_precision(monkeypatch):
    for cli_args, expected_dtype in (([], torch.bfloat16), (["--fp16"], torch.float16)):
        monkeypatch.setattr(sys, "argv", ["miles", *cli_args])
        args = load_fsdp_args()
        args.true_on_policy_mode = True

        assert args.bf16 == (not args.fp16)
        assert resolve_precision_policy(None, args).param_dtype is expected_dtype
        assert _rollout_logprob_dtype(args) is expected_dtype


def test_qwen3_formal_true_on_policy_resolves_fp32_params_with_bf16_autocast():
    args = SimpleNamespace(
        fp16=False,
        keep_fp32_master=True,
        true_on_policy_mode=True,
        sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
    )

    policy = resolve_precision_policy(SimpleNamespace(model_type="qwen3"), args)

    assert policy.param_dtype is torch.float32
    assert policy.reduce_dtype is torch.float32
    assert policy.autocast_dtype is torch.bfloat16
    assert policy.keep_fp32_master


@pytest.mark.parametrize(
    ("model_type", "true_on_policy_mode", "contract"),
    [
        ("qwen3", False, QWEN3_DENSE_TRUE_ON_POLICY_V1.name),
        ("qwen3", True, None),
        ("qwen3_moe", True, QWEN3_DENSE_TRUE_ON_POLICY_V1.name),
    ],
)
def test_qwen3_formal_precision_does_not_leak_to_other_modes(model_type, true_on_policy_mode, contract):
    policy = resolve_precision_policy(
        SimpleNamespace(model_type=model_type),
        SimpleNamespace(
            fp16=False,
            keep_fp32_master=True,
            true_on_policy_mode=true_on_policy_mode,
            sglang_true_on_policy_contract=contract,
        ),
    )

    assert policy.param_dtype is torch.bfloat16
    assert policy.autocast_dtype is None


def test_qwen3_formal_true_on_policy_rejects_fp16():
    with pytest.raises(ValueError, match="requires bf16 training"):
        resolve_precision_policy(
            SimpleNamespace(model_type="qwen3"),
            SimpleNamespace(
                fp16=True,
                keep_fp32_master=True,
                true_on_policy_mode=True,
                sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
            ),
        )


def test_qwen3_formal_true_on_policy_rejects_disabled_fp32_master():
    with pytest.raises(ValueError, match="requires fp32 master weights"):
        resolve_precision_policy(
            SimpleNamespace(model_type="qwen3"),
            SimpleNamespace(
                fp16=False,
                keep_fp32_master=False,
                true_on_policy_mode=True,
                sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
            ),
        )


def test_precision_forward_context_uses_policy_autocast(monkeypatch):
    entered = []

    @contextmanager
    def fake_autocast(*, device_type, dtype):
        entered.append((device_type, dtype))
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    policy = resolve_precision_policy(
        SimpleNamespace(model_type="qwen3"),
        SimpleNamespace(
            fp16=False,
            keep_fp32_master=True,
            true_on_policy_mode=True,
            sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
        ),
    )

    with precision_forward_context(policy):
        pass

    assert entered == [("cuda", torch.bfloat16)]


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
