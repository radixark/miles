"""Tests for the A_log fp32 preservation chain.

Feature: Qwen3.5's ``A_log`` must end up as fp32 in the Megatron parameter
after hf->mcore conversion, because the chunk-gated-delta-rule kernel relies
on that precision. Two complementary pieces keep this invariant:

- Downstream — ``enforce_marked_param_dtypes`` (this module):
    Megatron's ``Float16Module`` unconditionally casts every floating-point
    parameter to bf16/fp16 at wrap time. There is no declarative opt-out in
    nn.Module or Megatron; even Megatron's own MoE router uses the same
    post-hoc ``.data = ...to(float32)`` pattern in
    ``_maintain_float32_expert_bias``. We generalize that by letting model
    definitions declare intent via ``mark_param_dtype`` and re-casting after
    ``get_model`` returns.
- Upstream — ``Qwen3_5Bridge._weight_to_mcore_format``:
    mbridge's base ``_weight_to_mcore_format`` pre-casts every HF tensor to
    ``self.dtype`` (bf16) before TP scatter. For A_log that pre-cast rounds
    the fp32 HF value. The override returns A_log as fp32 early, bypassing
    that pre-cast entirely.

The end-to-end test ties both halves together and checks bit-exact equality
with the HF fp32 source — this is the regression guard against the original
``patch_weight_to_mcore_format_preserve_fp32`` failure mode, where only the
upstream cast was intercepted and the downstream ``t.to(param.dtype)`` in
``Bridge.load_weights`` still demoted A_log back to bf16.
"""

import pytest
import torch
import torch.nn as nn

from miles.backends.megatron_utils.fp32_param_utils import (
    FORCED_PARAM_DTYPE_ATTR,
    enforce_marked_param_dtypes,
    mark_param_dtype,
)


# ---------------------------------------------------------------------------
# Downstream: mark_param_dtype + enforce_marked_param_dtypes
# ---------------------------------------------------------------------------


class _ToyModule(nn.Module):
    """Minimal stand-in for Qwen3_5GatedDeltaNet: one marked fp32 param plus
    one regular bf16-target param, so we can check the collateral damage
    boundary of ``enforce_marked_param_dtypes``."""

    def __init__(self, num_heads: int = 8):
        super().__init__()
        A = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A).to(torch.float32))
        mark_param_dtype(self.A_log, torch.float32)
        self.in_proj = nn.Linear(16, num_heads, bias=False)


class TestMarkParamDtype:
    def test_attaches_expected_attribute(self):
        p = nn.Parameter(torch.zeros(4))
        mark_param_dtype(p, torch.float32)
        assert getattr(p, FORCED_PARAM_DTYPE_ATTR) is torch.float32

    def test_overwrites_previous_mark(self):
        p = nn.Parameter(torch.zeros(4))
        mark_param_dtype(p, torch.float32)
        mark_param_dtype(p, torch.float64)
        assert getattr(p, FORCED_PARAM_DTYPE_ATTR) is torch.float64


class TestEnforceMarkedParamDtypes:
    def test_recasts_marked_param_back_to_fp32_after_float16_wrap(self):
        """Simulates the full Megatron path: construct -> bfloat16() (what
        ``Float16Module(...)`` does) -> enforce. A_log must come out fp32."""
        m = _ToyModule()
        assert m.A_log.dtype == torch.float32

        # Simulate Float16Module(config, m) — module.bfloat16() in the ctor
        # demotes every floating param including the marked one.
        m.bfloat16()
        assert m.A_log.dtype == torch.bfloat16

        enforce_marked_param_dtypes([m])
        assert m.A_log.dtype == torch.float32

    def test_preserves_parameter_identity(self):
        """Optimizer and DDP bucket parameters by Python identity, set up
        AFTER ``enforce_marked_param_dtypes`` runs. If we re-bind via
        ``self.A_log = nn.Parameter(...)`` the id changes and the optimizer
        map breaks. We must only mutate ``.data``."""
        m = _ToyModule()
        m.bfloat16()
        before_id = id(m.A_log)
        before_param_obj = m.A_log

        enforce_marked_param_dtypes([m])

        assert id(m.A_log) == before_id
        assert m.A_log is before_param_obj

    def test_leaves_unmarked_params_alone(self):
        m = _ToyModule()
        m.bfloat16()
        assert m.in_proj.weight.dtype == torch.bfloat16

        enforce_marked_param_dtypes([m])
        assert m.in_proj.weight.dtype == torch.bfloat16

    def test_is_noop_when_already_target_dtype(self):
        """Idempotency — second call must not re-allocate or change anything.
        Guards against accidental double-work when the hook is called on
        both the training and conversion entrypoints in the same process."""
        m = _ToyModule()
        m.bfloat16()
        enforce_marked_param_dtypes([m])

        data_before = m.A_log.data
        updated = enforce_marked_param_dtypes([m])
        assert m.A_log.dtype == torch.float32
        # ``.data`` should be the same tensor object (no unnecessary realloc).
        assert m.A_log.data.data_ptr() == data_before.data_ptr()
        # Name is still reported even on the no-realloc path — this is by
        # design so the rank-0 log line reflects policy coverage, not churn.
        assert any(n.endswith("A_log") for n in updated)

    def test_walks_multiple_model_chunks(self):
        """``setup_model_and_optimizer`` passes a list of model chunks (for
        virtual pipeline parallelism). The helper must iterate all of them."""
        chunks = [_ToyModule(), _ToyModule()]
        for c in chunks:
            c.bfloat16()

        enforce_marked_param_dtypes(chunks)
        for c in chunks:
            assert c.A_log.dtype == torch.float32

    def test_returns_empty_when_no_marks(self):
        m = nn.Linear(4, 4)
        m.bfloat16()
        assert enforce_marked_param_dtypes([m]) == []


# ---------------------------------------------------------------------------
# Upstream: Qwen3_5Bridge._weight_to_mcore_format
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bridge_stub():
    """Build a ``Qwen3_5Bridge`` without invoking ``__init__`` — ``__init__``
    needs a real HF config. The A_log branch only reads ``self.dtype``, which
    we set directly, so skipping init is safe and lets this test stay
    CPU-only and dep-free."""
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge

    bridge = Qwen3_5Bridge.__new__(Qwen3_5Bridge)
    return bridge


class TestQwen3_5BridgeALogOverride:
    A_LOG_NAME = "decoder.layers.0.self_attention.linear_attn.A_log"

    def test_returns_fp32_when_bridge_dtype_is_bf16(self, bridge_stub):
        """The override must bypass mbridge's ``w.to(self.dtype)`` pre-cast
        that would otherwise round HF fp32 to bf16 here."""
        bridge_stub.dtype = torch.bfloat16
        hf_tensor = torch.randn(32, dtype=torch.float32)

        out = bridge_stub._weight_to_mcore_format(self.A_LOG_NAME, [hf_tensor])

        assert out.dtype == torch.float32
        assert torch.equal(out, hf_tensor)
        assert out.is_contiguous()

    def test_upcasts_when_hf_input_is_bf16(self, bridge_stub):
        """A_log arriving as bf16 (non-canonical ckpt) is still forced to
        fp32 — the invariant is the output dtype, not the input's."""
        bridge_stub.dtype = torch.bfloat16
        hf_tensor = torch.randn(32, dtype=torch.bfloat16)

        out = bridge_stub._weight_to_mcore_format(self.A_LOG_NAME, [hf_tensor])

        assert out.dtype == torch.float32

    def test_mtp_layer_a_log_also_matches(self, bridge_stub):
        """The override uses ``endswith`` so MTP-layer A_log
        (``mtp.layers.{idx}...``) also matches — MTP is a real Qwen3.5
        variant and must not silently skip the override."""
        bridge_stub.dtype = torch.bfloat16
        hf_tensor = torch.randn(32, dtype=torch.float32)

        out = bridge_stub._weight_to_mcore_format("mtp.layers.0.self_attention.linear_attn.A_log", [hf_tensor])
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# End-to-end: the two halves together, matching ``Bridge.load_weights``.
# ---------------------------------------------------------------------------


class TestALogLoadPathEndToEnd:
    """Replays the dtype-relevant subset of ``Bridge.load_weights`` on a toy
    model, as documented in ``tools/debug_a_log_old_flow.py``. No distributed
    or real safetensor IO — only the two cast points we care about.

    Expected outcome: HF fp32 value lands in the Megatron A_log param
    bit-exactly. Regression target: the OLD ``patch_weight_to_mcore_format_preserve_fp32``
    failed here because ``bridge.py:246`` still cast down to ``param.dtype == bf16``.
    """

    def test_lossless_roundtrip(self, bridge_stub):
        a_log_name = "decoder.layers.0.self_attention.linear_attn.A_log"
        hf_tensor = torch.randn(32, dtype=torch.float32)

        # 1. Build model (A_log marked fp32 at definition site).
        model = _ToyModule(num_heads=32)

        # 2. Megatron wraps with Float16Module → .bfloat16().
        model.bfloat16()

        # 3. enforce_marked_param_dtypes restores A_log to fp32 BEFORE
        #    load_weights runs, so ``param.dtype`` at bridge.py:246 is fp32.
        enforce_marked_param_dtypes([model])
        assert model.A_log.dtype == torch.float32

        # 4. mbridge: _weight_to_mcore_format (with override → fp32).
        bridge_stub.dtype = torch.bfloat16  # would demote without override
        mcore_weight = bridge_stub._weight_to_mcore_format(a_log_name, [hf_tensor])
        assert mcore_weight.dtype == torch.float32

        # 5. mbridge bridge.py:246 — ``t.to(param.device, dtype=param.dtype)``.
        param = model.A_log
        staged = mcore_weight.to(param.device, dtype=param.dtype).contiguous()
        assert staged.dtype == torch.float32  # no-op cast

        # 6. mbridge bridge.py:258 — ``param.copy_(param_to_load)``.
        param.data.copy_(staged)

        # Bit-exact round-trip: both halves were required to get here.
        assert model.A_log.dtype == torch.float32
        assert torch.equal(model.A_log.data, hf_tensor)

    def test_old_patch_only_regresses_without_enforce(self, bridge_stub):
        """Negative control: if we DROP ``enforce_marked_param_dtypes`` and
        only keep the upstream override (the shape of the old patch), the
        downstream ``t.to(param.dtype)`` still rounds to bf16. This pins the
        old failure mode so it cannot be re-introduced by accident."""
        a_log_name = "decoder.layers.0.self_attention.linear_attn.A_log"
        # Use a value where bf16 rounding is observable.
        hf_tensor = torch.tensor([0.970378123] * 8, dtype=torch.float32)

        model = _ToyModule(num_heads=8)
        model.bfloat16()  # A_log is bf16; no enforce call here on purpose.

        bridge_stub.dtype = torch.bfloat16
        mcore_weight = bridge_stub._weight_to_mcore_format(a_log_name, [hf_tensor])
        assert mcore_weight.dtype == torch.float32

        staged = mcore_weight.to(model.A_log.device, dtype=model.A_log.dtype).contiguous()
        # Regression check: demoted to bf16 because param.dtype is bf16.
        assert staged.dtype == torch.bfloat16
        assert not torch.equal(staged.to(torch.float32), hf_tensor)
