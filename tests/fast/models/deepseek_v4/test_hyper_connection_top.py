import pytest
import torch

from miles_plugins.models.deepseek_v4.ops import hyper_connection


def _inputs():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    hc_fn = torch.randn(4, 32, dtype=torch.float32)
    hc_scale = torch.randn(1, dtype=torch.float32)
    hc_base = torch.randn(4, dtype=torch.float32)
    return x, hc_fn, hc_scale, hc_base


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_top_fused_hc_head_flag_accepts_true_values(monkeypatch, value):
    monkeypatch.setenv(hyper_connection._TOP_FUSED_HC_HEAD_ENV, value)
    assert hyper_connection._top_fused_hc_head_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_top_fused_hc_head_flag_accepts_false_values(monkeypatch, value):
    monkeypatch.setenv(hyper_connection._TOP_FUSED_HC_HEAD_ENV, value)
    assert not hyper_connection._top_fused_hc_head_enabled()


def test_top_fused_hc_head_flag_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv(hyper_connection._TOP_FUSED_HC_HEAD_ENV, "maybe")
    with pytest.raises(ValueError, match="must be a boolean value"):
        hyper_connection._top_fused_hc_head_enabled()


def test_sglang_fused_hc_head_wrapper_uses_reference_backward(monkeypatch):
    x, hc_fn, hc_scale, hc_base = _inputs()
    values = [
        value.detach().requires_grad_(True)
        for value in (x, hc_fn, hc_scale, hc_base)
    ]
    reference_values = [
        value.detach().clone().requires_grad_(True)
        for value in (x, hc_fn, hc_scale, hc_base)
    ]

    def fake_fused_forward(
        x_value,
        fn_value,
        scale_value,
        base_value,
        norm_eps,
        hc_eps,
    ):
        return hyper_connection._hc_head_reference(
            x_value,
            fn_value,
            scale_value,
            base_value,
            norm_eps,
            hc_eps,
        )

    monkeypatch.setattr(
        hyper_connection,
        "_sglang_fused_hc_head_forward",
        fake_fused_forward,
    )

    norm_eps = 1e-6
    hc_eps = 1e-6
    actual = hyper_connection._SGLangFusedHCHead.apply(
        *values,
        norm_eps,
        hc_eps,
    )
    expected = hyper_connection._hc_head_reference(
        *reference_values,
        norm_eps,
        hc_eps,
    )
    assert torch.equal(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for actual_value, expected_value in zip(values, reference_values):
        torch.testing.assert_close(
            actual_value.grad,
            expected_value.grad,
            rtol=0,
            atol=0,
        )
