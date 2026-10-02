# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact tests for the fused CuTe DSL NVFP4 QDQ kernel.

The data patterns and Four Over Six Cartesian matrix mirror FlashInfer's
``tests/utils/test_fp4_quantize.py::test_nvfp4_quantize_te_reference``. The
oracle follows Transformer Engine's strict
``tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py`` test and calls TE's native
quantize-then-dequantize path because FlashInfer's per-tensor path has a
different numerical contract.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("cutlass")
te = pytest.importorskip("transformer_engine.pytorch")

from tests.ci.ci_register import register_cuda_ci

from miles.utils.fused_nvfp4_qdq import NVFP4QDQConfig, NVFP4QDQErrorMode
from miles.utils.fused_nvfp4_qdq import compute_nvfp4_amax as scalar_amax  # noqa: E402
from miles.utils.fused_nvfp4_qdq import fused_nvfp4_qdq as scalar_qdq
from miles.utils.grouped_nvfp4_qdq import compute_grouped_nvfp4_amax as compute_nvfp4_amax
from miles.utils.grouped_nvfp4_qdq import fake_grouped_nvfp4_quantization_ste as fake_nvfp4_quantization_ste
from miles.utils.grouped_nvfp4_qdq import fused_grouped_nvfp4_qdq as fused_nvfp4_qdq
from miles.utils.nvfp4_fake_qat import maybe_fake_quantize_nvfp4_weight_tensors

register_cuda_ci(est_time=60, suite="stage-c-8-gpu-b200", labels=["precision"], hardware=["blackwell"])

_recipe_available, _recipe_unavailable_reason = te.is_nvfp4_available(return_reason=True)
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(not _recipe_available, reason=_recipe_unavailable_reason),
]


@pytest.fixture(scope="module", autouse=True)
def _select_local_cuda_device():
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


SHAPES = [
    # Minimum-K and odd-row cases absent from FlashInfer's swizzled-layout matrix.
    (1, 16),
    (1, 32),
    (3, 48),
    # FlashInfer strict-test shapes and both TE BF16 dispatch routes after M padding.
    (1, 64),
    (3, 128),
    (16, 64),
    (31, 128),
    (32, 128),
    (128, 64),
    (128, 1024),
    (256, 256),
    (1024, 2048),
]
GROUP_COUNTS = [1, 3, 8]


CONFIGS = [
    pytest.param(NVFP4QDQConfig(), id="nvfp4"),
]
for _error_mode in (NVFP4QDQErrorMode.MAE, NVFP4QDQErrorMode.MSE):
    for _e4m3_max in (448, 256):
        for _error_use_fast_math in (False, True):
            CONFIGS.append(
                pytest.param(
                    NVFP4QDQConfig(
                        use_4over6=True,
                        e4m3_max=_e4m3_max,
                        error_mode=_error_mode,
                        error_use_fast_math=_error_use_fast_math,
                    ),
                    id=(
                        f"4over6-{_error_mode.name.lower()}-e4m3-{_e4m3_max}-"
                        f"{'fp16-error' if _error_use_fast_math else 'exact-error'}"
                    ),
                )
            )


def _make_input(
    group_count: int,
    shape: tuple[int, int],
    dtype: torch.dtype,
    init_data: str,
) -> torch.Tensor:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    m, n = shape
    if init_data == "random":
        x = torch.randn((group_count, m, n), dtype=torch.float32, device="cuda")
        if m > 1:
            x[:, 0].zero_()
        scales = torch.pow(
            2.0,
            (torch.arange(group_count, device="cuda") % 5) - 2,
        ).view(-1, 1, 1)
        return (x * scales).to(dtype)
    if init_data == "boundary":
        base = torch.linspace(-12.0, 12.0, steps=n // 2, dtype=torch.float32, device="cuda")
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, torch.full_like(base, 1e-4))
        row = torch.empty(n, dtype=torch.float32, device="cuda")
        row[0::2] = base - eps
        row[1::2] = base + eps
        groups = []
        for group_idx in range(group_count):
            scale = 2.0 ** ((group_idx % 5) - 2)
            groups.append(torch.roll(row, shifts=2 * group_idx).repeat(m, 1) * scale)
        return torch.stack(groups).to(dtype=dtype)
    if init_data == "zeros":
        # Alternate signed zeros so the integer-view equality below exercises
        # TE's E2M1 sign-bit contract for zero-amax blocks.
        row = torch.tensor([-0.0, 0.0], dtype=torch.float32, device="cuda").repeat(m, n // 2)
        return torch.stack([torch.roll(row, shifts=group_idx % 2, dims=1) for group_idx in range(group_count)]).to(
            dtype=dtype
        )
    if init_data == "maxes":
        x = torch.full((group_count, m, n), torch.finfo(dtype).max, dtype=dtype, device="cuda")
        signs = torch.where(
            torch.arange(group_count, device="cuda") % 2 == 0,
            1.0,
            -1.0,
        ).to(dtype)
        return x * signs.view(-1, 1, 1)
    raise ValueError(f"Unknown init_data: {init_data}")


def _make_te_quantizer(config: NVFP4QDQConfig):
    return te.NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=config.use_4over6,
        nvfp4_e4m3_max=config.e4m3_max,
        nvfp4_4over6_err_mode=config.error_mode.name,
        with_random_sign_mask=False,
    )


def _te_reference(x: torch.Tensor, config: NVFP4QDQConfig) -> tuple[torch.Tensor, torch.Tensor]:
    _, m, n = x.shape
    references = []
    amaxes = []
    quantizer = _make_te_quantizer(config)
    for weight in x.unbind(0):
        padded_m = ((m + 15) // 16) * 16
        if padded_m == m:
            weight_padded = weight.contiguous()
        else:
            padding = torch.zeros((padded_m - m, n), dtype=x.dtype, device=x.device)
            weight_padded = torch.cat((weight.contiguous(), padding), dim=0)

        quantized = quantizer.quantize(weight_padded)
        references.append(quantized.dequantize(dtype=x.dtype)[:m, :n].contiguous())
        assert quantized._amax_rowwise is not None
        amaxes.append(quantized._amax_rowwise.reshape(1))
    return torch.stack(references), torch.cat(amaxes)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("group_count", GROUP_COUNTS, ids=lambda value: f"g{value}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"{shape[0]}x{shape[1]}")
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("config", CONFIGS)
@torch.inference_mode()
def test_fused_nvfp4_qdq_is_bit_exact_with_te(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    group_count: int,
    shape: tuple[int, int],
    init_data: str,
    config: NVFP4QDQConfig,
) -> None:
    """Cover BF16/FP16 x shapes x data patterns x the full supported feature matrix."""
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if config.error_use_fast_math else "0")
    x = _make_input(group_count, shape, dtype, init_data)
    amaxes = compute_nvfp4_amax(x)
    expected, te_amax = _te_reference(x, config)
    actual = fused_nvfp4_qdq(x, amaxes, config)

    assert torch.equal(amaxes.view(torch.int32), te_amax.view(torch.int32))
    # Integer views distinguish signed zero; tolerance-zero floating comparison does not.
    actual_bits = actual.view(torch.uint16)
    expected_bits = expected.view(torch.uint16)
    assert torch.equal(
        actual_bits, expected_bits
    ), f"bit mismatch count: {torch.count_nonzero(actual_bits != expected_bits).item()}"
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    # Also protect the existing rank-2 path when sharing the store helper.
    loop = torch.stack([scalar_qdq(w, scalar_amax(w), config) for w in x.unbind(0)])
    assert torch.equal(actual_bits, loop.view(torch.uint16))


@pytest.mark.parametrize("group_count", [1, 8], ids=lambda value: f"g{value}")
def test_fused_nvfp4_qdq_uses_straight_through_gradient_and_preserves_main_grad(
    group_count: int,
) -> None:
    x = torch.nn.Parameter(torch.randn((group_count, 3, 32), dtype=torch.bfloat16, device="cuda"))
    main_grad = torch.empty_like(x)
    x.main_grad = main_grad
    output = fake_nvfp4_quantization_ste(x, NVFP4QDQConfig())
    grad = (
        torch.arange(1, group_count + 1, dtype=torch.float32, device="cuda")
        .view(-1, 1, 1)
        .expand_as(output)
        .to(output.dtype)
    )
    output.backward(grad)

    assert tuple(output.shape) == tuple(x.shape)
    assert output.is_contiguous()
    torch.testing.assert_close(x.grad, grad, rtol=0.0, atol=0.0)
    assert output.main_grad is main_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_nvfp4_qdq_rejects_unsupported_input_dtype(dtype: torch.dtype) -> None:
    x = torch.randn((1, 2, 16), dtype=dtype, device="cuda")
    with pytest.raises(TypeError, match="supports BF16 and FP16"):
        fused_nvfp4_qdq(x, torch.ones(1, dtype=torch.float32, device="cuda"))


def test_fused_nvfp4_qdq_rejects_rank_2_input() -> None:
    x = torch.randn((2, 16), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="rank-3"):
        fused_nvfp4_qdq(x, torch.ones(1, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("group_count", [0, 2049])
def test_fused_nvfp4_qdq_rejects_group_count_outside_bound(
    group_count: int,
) -> None:
    x = torch.empty((group_count, 1, 16), dtype=torch.bfloat16, device="cuda")
    amaxes = torch.empty(group_count, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="1 <= G <= 2048"):
        fused_nvfp4_qdq(x, amaxes, NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_non_block_aligned_k() -> None:
    x = torch.randn((3, 2, 17), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="N divisible by 16"):
        fused_nvfp4_qdq(x, compute_nvfp4_amax(x), NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_misaligned_contiguous_storage() -> None:
    storage = torch.randn(33, dtype=torch.bfloat16, device="cuda")
    x = storage[1:].view(1, 2, 16)
    assert x.is_contiguous()
    assert x.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte-aligned"):
        fused_nvfp4_qdq(x, compute_nvfp4_amax(x), NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_wrong_amax_shape() -> None:
    x = torch.randn((3, 2, 16), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        fused_nvfp4_qdq(x, torch.ones(1, dtype=torch.float32, device="cuda"), NVFP4QDQConfig())


@torch.inference_mode()
def test_fused_nvfp4_qdq_supports_maximum_group_count() -> None:
    config = NVFP4QDQConfig()
    x = _make_input(2048, (1, 16), torch.bfloat16, "random")
    amaxes = compute_nvfp4_amax(x)
    expected, te_amaxes = _te_reference(x, config)
    actual = fused_nvfp4_qdq(x, amaxes, config)

    assert torch.equal(amaxes.view(torch.int32), te_amaxes.view(torch.int32))
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fused_nvfp4_qdq_uses_and_restores_non_current_device() -> None:
    with torch.cuda.device(0):
        with torch.cuda.device(1):
            x = _make_input(3, (3, 32), torch.bfloat16, "boundary")
            amaxes = compute_nvfp4_amax(x)
            expected, _ = _te_reference(x, NVFP4QDQConfig())

        assert torch.cuda.current_device() == 0
        actual = fused_nvfp4_qdq(x, amaxes, NVFP4QDQConfig())
        assert torch.cuda.current_device() == 0

    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


@pytest.fixture
def qat_env(monkeypatch):
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    monkeypatch.setenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "1")
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "none")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "0")


def _native_layer(fuse_wgrad=False):
    import inspect

    if "use_grouped_tensor" not in inspect.signature(te.GroupedLinear).parameters:
        pytest.skip("Native packed TE GroupedLinear is required")

    class QATGroupedLinear(te.GroupedLinear):
        # Same hook used by the Miles Megatron fork.
        def _get_weight_tensors(self):
            return maybe_fake_quantize_nvfp4_weight_tensors(super()._get_weight_tensors())

    return QATGroupedLinear(
        3,
        128,
        64,
        bias=False,
        params_dtype=torch.bfloat16,
        single_grouped_weight=True,
        use_grouped_tensor=True,
        fuse_wgrad_accumulation=fuse_wgrad,
    )


@pytest.mark.usefixtures("qat_env")
@pytest.mark.parametrize("use_4over6", [False, True])
@pytest.mark.parametrize("fuse_wgrad", [False, True])
def test_native_te_forward_backward_and_update(monkeypatch, use_4over6, fuse_wgrad):
    from miles.utils.fused_nvfp4_qdq import current_nvfp4_qdq_config

    if use_4over6:
        monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MSE")
    layer = _native_layer(fuse_wgrad)
    weight = layer.weight
    assert list(dict(layer.named_parameters())) == ["weight"]
    original = weight.rowwise_data.view(3, 64, 128).clone()
    original_ptr = weight.rowwise_data.data_ptr()
    if fuse_wgrad:
        weight.main_grad = torch.zeros_like(original, dtype=torch.float32)
    qweight = layer._get_weight_tensors()[0]
    assert qweight.requires_grad and qweight.grad_fn is not None
    assert qweight.rowwise_data.data_ptr() != original_ptr
    if fuse_wgrad:
        assert qweight.main_grad is weight.main_grad
    expected, _ = _te_reference(original, current_nvfp4_qdq_config())
    assert torch.equal(qweight.rowwise_data.view(3, 64, 128).view(torch.uint16), expected.view(torch.uint16))
    assert torch.equal(original, weight.rowwise_data.view_as(original))

    # Include a zero-token expert and different splits to catch expert reordering.
    splits = [2, 0, 3]
    inputs = torch.cat(
        [torch.full((tokens, 128), i + 1, device="cuda", dtype=torch.bfloat16) for i, tokens in enumerate(splits)]
    ).requires_grad_()
    output = layer(inputs, torch.tensor(splits, device="cuda", dtype=torch.int64))
    reference = torch.cat(
        [torch.nn.functional.linear(inp, w) for inp, w in zip(inputs.split(splits), expected, strict=True)]
    )
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.01)
    output.backward(torch.ones_like(output))
    expected_grad = torch.stack([torch.full_like(original[0], n * (i + 1)) for i, n in enumerate(splits)])
    grad = weight.main_grad if fuse_wgrad else weight.grad
    torch.testing.assert_close(grad, expected_grad.to(grad.dtype), rtol=0, atol=0)
    expected_dgrad = torch.cat(
        [w.float().sum(0).to(inputs.dtype).expand(n, -1) for w, n in zip(expected, splits, strict=True)]
    )
    torch.testing.assert_close(inputs.grad, expected_dgrad, rtol=0.02, atol=0.01)

    # Fused accumulation writes main_grad; the optimizer consumes that buffer.
    if fuse_wgrad:
        weight.grad = weight.main_grad.to(weight.dtype)
    optimizer = torch.optim.SGD([weight], lr=0.125)
    optimizer.step()
    assert layer.weight is weight and weight.rowwise_data.data_ptr() == original_ptr
    torch.testing.assert_close(weight.rowwise_data.view_as(original), original - expected_grad * 0.125, rtol=0, atol=0)


@pytest.mark.usefixtures("qat_env")
@pytest.mark.parametrize("native", [False, True])
def test_graph_replay_recomputes_each_expert_amax(native):
    if native:
        weight = _native_layer().weight
        storage = weight.rowwise_data.view(3, 64, 128)
    else:
        weight = torch.nn.Parameter(torch.randn((3, 64, 128), device="cuda", dtype=torch.bfloat16))
        storage = weight.detach()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            maybe_fake_quantize_nvfp4_weight_tensors([weight])
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = maybe_fake_quantize_nvfp4_weight_tensors([weight])[0]
    result = output.rowwise_data.view_as(storage) if native else output
    for scale in (0.01, 100.0):
        with torch.no_grad():
            storage[1].normal_().mul_(scale)
            storage[2].zero_()
        graph.replay()
        expected, _ = _te_reference(storage, NVFP4QDQConfig())
        assert torch.equal(result.view(torch.uint16), expected.view(torch.uint16))


@pytest.mark.usefixtures("qat_env")
def test_native_adapter_rejects_irregular_layout():
    weight = _native_layer().weight
    weight.offsets = [0, 0, 0]
    with pytest.raises(ValueError, match="uniform, unquantized, densely ordered"):
        maybe_fake_quantize_nvfp4_weight_tensors([weight])


@pytest.mark.usefixtures("qat_env")
def test_native_checkpoint_keeps_original_high_precision_weights(tmp_path):
    layer = _native_layer()
    original = layer.weight.rowwise_data.clone()
    layer._get_weight_tensors()
    checkpoint = tmp_path / "grouped.pt"
    torch.save(layer.state_dict(), checkpoint)
    restored = _native_layer()
    with torch.serialization.safe_globals([type(layer.weight)]):
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(restored.weight.rowwise_data, original)
    actual = restored._get_weight_tensors()[0].rowwise_data
    expected = layer._get_weight_tensors()[0].rowwise_data
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


def test_grouped_rejects_noncontiguous_input_and_amax():
    x = torch.randn((3, 32, 32), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous"):
        fused_nvfp4_qdq(x.transpose(1, 2), compute_nvfp4_amax(x), NVFP4QDQConfig())
    with pytest.raises(ValueError, match="contiguous"):
        fused_nvfp4_qdq(x, torch.ones(6, device="cuda")[::2], NVFP4QDQConfig())


@pytest.mark.usefixtures("qat_env")
@pytest.mark.parametrize("overwrite", [False, True])
def test_native_fused_wgrad_reaches_megatron_leaf_hook(overwrite):
    layer = _native_layer(fuse_wgrad=True)
    weight = layer.weight
    weight.main_grad = torch.ones((3, 64, 128), device="cuda", dtype=torch.float32)
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = True
    weight.overwrite_main_grad = overwrite
    weight.get_main_grad = lambda: weight.main_grad
    hook_calls = []

    def ddp_hook(param):
        # Megatron uses this marker to skip a second add into main_grad, and
        # the hook itself marks the parameter ready for gradient reduction.
        assert param.grad_added_to_main_grad
        assert torch.count_nonzero(param.grad).item() == 0
        hook_calls.append(True)
        param.grad = None

    weight.register_post_accumulate_grad_hook(ddp_hook)
    splits = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int64)
    inputs = torch.ones((5, 128), device="cuda", dtype=torch.bfloat16)
    grad = torch.stack([torch.full((64, 128), n, device="cuda", dtype=torch.float32) for n in (2, 0, 3)])
    for step in range(2):
        weight.grad_added_to_main_grad = False
        layer(inputs, splits).sum().backward()
        assert len(hook_calls) == step + 1
        expected = grad if overwrite else 1 + (step + 1) * grad
        torch.testing.assert_close(weight.main_grad, expected, rtol=0, atol=0)
