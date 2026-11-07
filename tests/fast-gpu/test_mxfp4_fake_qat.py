from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

from miles.kernels.quant.fake_quant import fake_quant_fp4_forward
from miles.utils.mxfp4 import E2M1_VALUES, MXFP4_GROUP_SIZE, mxfp4_quantize
from miles.utils.mxfp4_fake_qat import fake_quantize_mxfp4_weight_tensors

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)


def _dequantize_reference(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logical_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
    codes = packed.view(torch.uint8)
    nibbles = torch.stack((codes & 0x0F, (codes >> 4) & 0x0F), dim=-1).flatten(-2)
    table = torch.tensor(E2M1_VALUES + tuple(-value for value in E2M1_VALUES), device=packed.device)
    values = table[nibbles.long()].reshape(-1, MXFP4_GROUP_SIZE)
    scale_factor = torch.exp2(scale.view(torch.uint8).float() - 127).reshape(-1, 1)
    return (values * scale_factor).reshape(logical_shape)


def _rollout_weight(weight: torch.Tensor) -> torch.Tensor:
    """The weight a rollout decodes from the MXFP4 encoding of ``weight``."""
    packed, scale = mxfp4_quantize(weight)
    return _dequantize_reference(packed, scale).to(weight.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fake_quant_matches_rollout_codec(dtype):
    torch.manual_seed(0)
    magnitudes = torch.logspace(-30, 30, steps=67, base=2.0, device="cuda").reshape(-1, 1)
    weight = (torch.randn((67, 4 * MXFP4_GROUP_SIZE), device="cuda") * magnitudes).to(dtype)
    original = weight.clone()

    assert torch.equal(fake_quant_fp4_forward(weight), _rollout_weight(weight))
    assert torch.equal(weight, original)


def _boundary_blocks() -> torch.Tensor:
    """Blocks whose amax/6 sits on or next to a power of two, with values on and next to rounding ties."""
    blocks = []
    for exponent in (-140, -130, -126, -24, -9, 0, 3, 8):
        scale = 2.0**exponent
        for amax_factor in (1.0, 1.0 + 2.0**-23, 1.0 - 2.0**-24):
            for tie in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
                block = torch.full((MXFP4_GROUP_SIZE,), tie * scale, dtype=torch.float32)
                block[0] = 6.0 * scale * amax_factor
                block[1::2] *= -1
                block[2] = torch.nextafter(block[2], torch.tensor(float("inf")))
                block[3] = torch.nextafter(block[3], torch.tensor(0.0))
                blocks.append(block)
    return torch.stack(blocks).cuda()


def test_fake_quant_matches_rollout_codec_at_scale_and_rounding_boundaries():
    weight = _boundary_blocks()

    assert torch.equal(fake_quant_fp4_forward(weight), _rollout_weight(weight))


def test_fake_quant_rounds_in_place():
    torch.manual_seed(1)
    weight = torch.randn((8, 4 * MXFP4_GROUP_SIZE), device="cuda", dtype=torch.bfloat16)
    expected = fake_quant_fp4_forward(weight)
    data_ptr = weight.data_ptr()

    fake_quant_fp4_forward(weight, out=weight)

    assert torch.equal(weight, expected)
    assert weight.data_ptr() == data_ptr


def test_rounding_is_idempotent_in_value():
    """Rounded weights round to themselves, and a rollout decodes them unchanged from their re-encoding."""
    torch.manual_seed(2)
    rounded = fake_quant_fp4_forward(torch.randn((64, 4 * MXFP4_GROUP_SIZE), device="cuda", dtype=torch.bfloat16))

    assert torch.equal(fake_quant_fp4_forward(rounded), rounded)
    assert torch.equal(_rollout_weight(rounded), rounded)


@pytest.mark.parametrize(
    ("linear_name", "shape"),
    [
        ("linear_fc1", (6 * MXFP4_GROUP_SIZE, 4 * MXFP4_GROUP_SIZE)),
        ("linear_fc2", (4 * MXFP4_GROUP_SIZE, 3 * MXFP4_GROUP_SIZE)),
    ],
)
def test_rounded_grid_matches_rollout_after_native_layout_conversion(linear_name, shape):
    torch.manual_seed(1)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def convert_native_layout(value):
        return value.chunk(2, dim=0) if linear_name == "linear_fc1" else (value,)

    qat_converted = convert_native_layout(fake_quant_fp4_forward(weight))
    rollout_converted = convert_native_layout(weight)

    for qat_weight, rollout_weight in zip(qat_converted, rollout_converted, strict=True):
        assert torch.equal(qat_weight, _rollout_weight(rollout_weight))


def test_adapter_rounds_the_parameters_in_place_on_the_first_microbatch_only():
    torch.manual_seed(4)
    weight = torch.nn.Parameter(torch.randn((4, 2 * MXFP4_GROUP_SIZE), device="cuda", dtype=torch.bfloat16))
    original = weight.detach().clone()
    module = SimpleNamespace(is_first_microbatch=False)

    assert fake_quantize_mxfp4_weight_tensors(module, [weight])[0] is weight
    assert torch.equal(weight.detach(), original)

    module.is_first_microbatch = True
    assert fake_quantize_mxfp4_weight_tensors(module, [weight])[0] is weight
    assert torch.equal(weight.detach(), fake_quant_fp4_forward(original))
    assert weight.is_leaf and weight.requires_grad


def test_adapter_rejects_megatron_fsdp_weights():
    weight = torch.randn((4, 2 * MXFP4_GROUP_SIZE), device="cuda", dtype=torch.bfloat16)
    weight.__fsdp_param__ = True

    with pytest.raises(RuntimeError, match="FSDP"):
        fake_quantize_mxfp4_weight_tensors(SimpleNamespace(is_first_microbatch=True), [weight])


def _megatron_ddp_hook(param):
    """Megatron DDP's backward post-hook for one parameter."""
    if param.grad is not None and (not param.grad_added_to_main_grad or getattr(param, "zero_out_wgrad", False)):
        param.main_grad.add_(param.grad.data)
    param.grad = None


def _grouped_linear(te, *, fake_quant: bool, fuse_wgrad_accumulation: bool):
    """A TE GroupedLinear, optionally with the fake-quantization hook of Megatron's TEGroupedLinear."""
    base = te.GroupedLinear

    class _FakeQuantGroupedLinear(base):
        is_first_microbatch = True

        def _get_weight_tensors(self):
            return fake_quantize_mxfp4_weight_tensors(self, super()._get_weight_tensors())

    cls = _FakeQuantGroupedLinear if fake_quant else base
    return cls(
        2,
        4 * MXFP4_GROUP_SIZE,
        64,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    )


@pytest.mark.parametrize("fuse_wgrad_accumulation", [False, True])
def test_te_grouped_linear_trains_on_rounded_weights_and_refreshes_them_each_pass(fuse_wgrad_accumulation):
    te = pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(3)
    m_splits = [24, 40]
    module = _grouped_linear(te, fake_quant=True, fuse_wgrad_accumulation=fuse_wgrad_accumulation)
    reference = _grouped_linear(te, fake_quant=False, fuse_wgrad_accumulation=fuse_wgrad_accumulation)
    params = list(te.GroupedLinear._get_weight_tensors(module))
    ref_params = list(reference._get_weight_tensors())
    masters = [param.detach().float().clone() for param in params]
    for param in params + ref_params:
        param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device="cuda")
        param.grad_added_to_main_grad = False
        param.register_post_accumulate_grad_hook(_megatron_ddp_hook)

    for _ in range(2):
        # The optimizer step writes the fp32 master weights into the BF16 parameters before each pass.
        with torch.no_grad():
            for param, ref_param, master in zip(params, ref_params, masters, strict=True):
                param.copy_(master)
                ref_param.copy_(fake_quant_fp4_forward(master.to(torch.bfloat16)))
                param.main_grad.zero_()
                ref_param.main_grad.zero_()
        module.is_first_microbatch = True
        for _ in range(2):
            inp = torch.randn((sum(m_splits), 4 * MXFP4_GROUP_SIZE), device="cuda", dtype=torch.bfloat16)
            grad_out = torch.randn((sum(m_splits), 64), device="cuda", dtype=torch.bfloat16)
            out = module(inp, m_splits)
            module.is_first_microbatch = False
            ref_out = reference(inp, m_splits)
            assert torch.equal(out, ref_out)
            out.backward(grad_out)
            ref_out.backward(grad_out)

        for param, ref_param, master in zip(params, ref_params, masters, strict=True):
            assert torch.equal(param.detach(), ref_param.detach())
            assert param.grad is None
            assert torch.equal(param.main_grad, ref_param.main_grad)
            master.sub_(0.05 * param.main_grad)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10, reason="MXFP8 needs Blackwell"
)
def test_mxfp8_grouped_linear_recomputes_with_the_rounded_weights():
    """With full recompute the first forward runs without grad and caches only rowwise MXFP8 weights."""
    te = pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.common.recipe import MXFP8BlockScaling

    torch.manual_seed(5)
    m_splits = [32, 64]
    module = _grouped_linear(te, fake_quant=True, fuse_wgrad_accumulation=False)
    reference = _grouped_linear(te, fake_quant=False, fuse_wgrad_accumulation=False)
    with torch.no_grad():
        for param, ref_param in zip(
            te.GroupedLinear._get_weight_tensors(module), reference._get_weight_tensors(), strict=True
        ):
            ref_param.copy_(fake_quant_fp4_forward(param.detach()))
    inp = torch.randn((sum(m_splits), 4 * MXFP4_GROUP_SIZE), device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn((sum(m_splits), 64), device="cuda", dtype=torch.bfloat16)

    outputs, input_grads, weight_grads = [], [], []
    for linear in (module, reference):
        linear.is_first_microbatch = True
        with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
            with torch.no_grad():
                linear(inp, m_splits, is_first_microbatch=True)
            linear.is_first_microbatch = False
            recompute_inp = inp.clone().requires_grad_()
            out = linear(recompute_inp, m_splits, is_first_microbatch=False)
        out.backward(grad_out)
        outputs.append(out)
        input_grads.append(recompute_inp.grad)
        weight_grads.append([weight.grad for weight in te.GroupedLinear._get_weight_tensors(linear)])

    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(input_grads[0], input_grads[1])
    for grad, ref_grad in zip(*weight_grads, strict=True):
        assert torch.equal(grad, ref_grad)
