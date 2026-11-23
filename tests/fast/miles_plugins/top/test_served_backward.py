"""TOP's FP8 backward is evaluated at the served weights (the processed bytes' dequantization), straight through to the
BF16 masters; KV caches are rounded in backward as serving stores them."""

import contextlib

import pytest
import torch

from miles_plugins.top import fp8_masters
from miles_plugins.top.fp8_forward import _StraightThrough, dequantize_processed
from miles_plugins.top.fp8_masters import FP8Master, dequantize_block_fp8
from miles_plugins.top.kernels import surrogate
from miles_plugins.top.kernels.surrogate import fp8_group_fake_quant, linear_surrogate


@pytest.fixture(autouse=True)
def _no_batch_invariant_mode(monkeypatch):
    # reference_mode switches Megatron's batch-invariant kernels off; nothing to switch on CPU.
    monkeypatch.setattr(surrogate, "reference_mode", contextlib.nullcontext)


def _checkpoint_pair(n, k, seed=0):
    """An E4M3 payload with power-of-two 128x128 block scales, as DeepSeek-V4 ships it."""
    g = torch.Generator().manual_seed(seed)
    payload = (torch.randn(n, k, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    exponent = torch.randint(-14, -6, ((n + 127) // 128, (k + 127) // 128), generator=g)
    return payload, torch.exp2(exponent.float())


def _pack_rows(scale, n):
    """Serving's processed scale layout: UE8M0 exponent bytes, four per int32 word (group 4w + j in byte j), one row
    of words per weight row."""
    exponent = (scale.view(torch.int32) >> 23) & 255
    exponent = torch.nn.functional.pad(exponent, (0, -exponent.shape[-1] % 4)).view(*exponent.shape[:-1], -1, 4)
    words = sum(exponent[..., j].to(torch.int64) << (8 * j) for j in range(4))
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return words.repeat_interleave(128, -2)[..., :n, :]


def test_dequantize_processed_packed_rows_is_the_exact_dequantization():
    payload, scale = _checkpoint_pair(320, 640)
    expected = dequantize_block_fp8(payload, scale)
    assert torch.equal(dequantize_processed(payload, _pack_rows(scale, 320)), expected)
    # FP32 block scales (the Hopper wo_a layout) decode the same way.
    assert torch.equal(dequantize_processed(payload, scale), expected)


def test_dequantize_processed_stacked_experts():
    pairs = [_checkpoint_pair(256, 384, seed=s) for s in range(3)]
    payload = torch.stack([p for p, _ in pairs])
    packed = torch.stack([_pack_rows(s, 256) for _, s in pairs])
    expected = torch.stack([dequantize_block_fp8(p, s) for p, s in pairs])
    assert torch.equal(dequantize_processed(payload, packed), expected)


def test_dequantize_processed_matches_sglangs_inverse_transform():
    fp8_utils = pytest.importorskip("sglang.srt.layers.quantization.fp8_utils")
    inverse = getattr(fp8_utils, "_inverse_transform_scale_ue8m0_impl", None)
    if inverse is None:
        pytest.skip("this SGLang has no inverse UE8M0 transform")
    payload, scale = _checkpoint_pair(256, 1024, seed=4)
    packed = _pack_rows(scale, 256)
    decoded = inverse(packed)[:, : scale.shape[-1]]  # one row per 128-row block
    assert torch.equal(dequantize_processed(payload, packed), dequantize_block_fp8(payload, decoded))


@pytest.mark.parametrize("case", ["fp4-experts", "1x128-scales", "32x32-scales"])
def test_dequantize_block_fp8_refuses_other_formats(case):
    """These broadcast into a wrong BF16 weight (and pass the power-of-two exactness check) without the refusal."""
    payload, scale = _checkpoint_pair(256, 256)
    if case == "fp4-experts":
        # DeepSeek-V4's FP4 experts: int8 [n, k/2] packed codes with [n, k/32] scales.
        payload, scale = payload.view(torch.int8)[:, :128], torch.ones(256, 8)
    elif case == "1x128-scales":
        scale = torch.ones(256, 2)
    else:
        scale = torch.ones(8, 8)
    with pytest.raises(ValueError, match=r"\[top\].*128x128 block scales"):
        dequantize_block_fp8(payload, scale)


def test_dequantize_processed_refuses_other_layouts():
    payload, scale = _checkpoint_pair(256, 256)
    with pytest.raises(ValueError, match="not per-row"):
        dequantize_processed(payload, _pack_rows(scale, 256)[::128])
    with pytest.raises(ValueError, match="no served value"):
        dequantize_processed(payload, scale.bfloat16())
    with pytest.raises(ValueError, match="not 128x128 blocks"):
        dequantize_processed(payload, scale[:1])


def test_straight_through_returns_the_served_bytes():
    master = torch.randn(64, 64, dtype=torch.bfloat16, requires_grad=True)
    served = (master.detach() * 1.07).bfloat16()
    out = _StraightThrough.apply(served, master)
    assert torch.equal(out, served)
    out.backward(torch.full_like(out, 3.0))
    assert torch.equal(master.grad, torch.full_like(master, 3.0))


def test_fp8_master_jacobian_weight_is_served_with_master_gradient():
    payload, scale = _checkpoint_pair(256, 256)
    fp8 = FP8Master(payload, scale, lambda w, s: (w, s), lambda p: dequantize_block_fp8(*p))
    served = dequantize_block_fp8(payload, scale)
    # The off-grid offset moves the master off the served values; the jacobian weight is the served one.
    assert not torch.equal(fp8.master.detach(), served)
    weight = fp8.jacobian_weight()
    assert torch.equal(weight, served)
    weight.sum().backward()
    assert torch.equal(fp8.master.grad, torch.ones_like(fp8.master))


def test_every_fp8_master_user_serves_its_weight(monkeypatch):
    """The three FP8Master users each hand it a value function that takes what their process returns: a processed pair
    (TrainableFP8Linear, DeepSeek-V4's FP8 wo_a) or one dequantized tensor (AbsorbedFP8Weight)."""
    from miles_plugins.top import fp8_forward
    from miles_plugins.top.dsv4_attention import _wo_a_value

    payload, scale = _checkpoint_pair(256, 256)
    expected = dequantize_block_fp8(payload, scale)

    # Serving's packing needs a GPU; the pair it returns is what the value function receives either way.
    monkeypatch.setattr(fp8_forward, "prepare_weight", lambda w, s: (w, s))
    monkeypatch.setattr(fp8_forward, "synced_weight", lambda w, s: (w, s))
    linear = fp8_forward.TrainableFP8Linear(payload, scale)
    assert torch.equal(linear.served(), expected)

    absorbed = fp8_forward.AbsorbedFP8Weight(payload, scale)
    assert torch.equal(absorbed.weight, absorbed._fp8.served())
    absorbed.weight.sum().backward()
    assert torch.equal(absorbed._fp8.master.grad, torch.ones_like(absorbed._fp8.master))

    rank, groups = 128, 2
    grouped = lambda w, s: (w, s.view(groups, -1, s.shape[-1]))  # noqa: E731
    wo_a = FP8Master(payload, scale, grouped, lambda pair: _wo_a_value(pair, o_lora_rank=rank))
    assert torch.equal(wo_a.served(), expected)


def test_linear_surrogate_backward_at_the_value():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 32, generator=g, requires_grad=True)
    master = torch.randn(16, 32, generator=g, requires_grad=True)
    served = master.detach() + 0.05 * torch.randn(16, 32, generator=g)
    out = linear_surrogate(lambda x, m: x @ served.t(), x, master, value_fn=lambda: served)
    grad = torch.randn(8, 16, generator=g)
    out.backward(grad)
    assert torch.allclose(x.grad, grad @ served)
    # dW of a linear does not depend on the weight; it goes to the master.
    assert torch.allclose(master.grad, grad.t() @ x.detach())

    x.grad = master.grad = None
    linear_surrogate(lambda x, m: x @ m.t(), x, master).backward(grad)
    assert torch.allclose(x.grad, grad @ master.detach())


def test_linear_surrogate_skips_the_value_without_input_gradient():
    x = torch.randn(4, 8)
    master = torch.randn(3, 8, requires_grad=True)

    def value():
        raise AssertionError("the served weight is decoded only for an input gradient")

    linear_surrogate(lambda x, m: x @ m.t(), x, master, value_fn=value).sum().backward()
    assert torch.allclose(master.grad, torch.ones(4, 3).t() @ x)


def test_served_from_checkpoint_is_the_sync_bytes_of_the_master():
    pytest.importorskip("sglang.srt.layers.quantization.fp8_utils")
    payload, scale = _checkpoint_pair(256, 384, seed=3)
    served = fp8_masters.served_from_checkpoint(payload, scale)
    # Power-of-two scales: the checkpoint's own values. FP32 scales: the UE8M0 requantization of them.
    assert torch.equal(served, dequantize_block_fp8(payload, scale))
    fp32_scale = scale * 1.3
    requantized = fp8_masters.served_from_checkpoint(payload, fp32_scale)
    master = fp8_masters.master_from_checkpoint(payload, fp32_scale)
    assert torch.equal(requantized, dequantize_block_fp8(*fp8_masters.quantize_checkpoint_format(master)))
    assert not torch.equal(requantized, dequantize_block_fp8(payload, fp32_scale))


def test_fp8_group_fake_quant_amax_is_the_dsa_cache_rule():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(6, 512, generator=g)
    x[2, :128] = 0
    groups = x.view(-1, 128)
    s = groups.abs().amax(-1, keepdim=True) / 448.0
    expected = torch.where(s > 0, (groups / s).to(torch.float8_e4m3fn).float() * s, 0.0).view_as(x)
    leaf = x.clone().requires_grad_()
    out = fp8_group_fake_quant(leaf, 128, scale="amax")
    assert torch.equal(out, expected) and not out.isnan().any()
    out.backward(torch.full_like(out, 2.0))
    assert torch.equal(leaf.grad, torch.full_like(x, 2.0))
    # The power-of-two rule rounds differently.
    assert not torch.equal(fp8_group_fake_quant(x, 128), expected)
    with pytest.raises(ValueError, match="unknown FP8 scale rule"):
        fp8_group_fake_quant(x, 128, scale="fp32")


@pytest.mark.parametrize("closed_form", [False, True])
@pytest.mark.parametrize("rows", [6, 20])
def test_row_surrogate_releases_its_reference_closures_after_backward(closed_form, rows):
    """The closures hold the decoded served experts; the graph can outlive the backward that needed them."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(rows, 4, generator=g, requires_grad=True)
    w = torch.randn(3, 4, generator=g, requires_grad=True)

    def reference(lo, hi, x_rows, weight):
        return x_rows @ weight.t()

    def closed(lo, hi, x_rows, grad, weight):
        return grad @ weight, [grad.t() @ x_rows]

    out = surrogate.row_surrogate(
        lambda x_rows, weight: x_rows @ weight.t(),
        reference,
        x,
        w,
        chunk_rows=8,
        chunk_backward_fn=closed if closed_form else None,
    )
    out.backward(torch.ones_like(out))
    assert torch.allclose(x.grad, torch.ones(rows, 3) @ w.detach())
    assert torch.allclose(w.grad, torch.ones(3, rows) @ x.detach())
    assert out.grad_fn.chunk_reference_fn is None and out.grad_fn.chunk_backward_fn is None
