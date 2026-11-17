"""Exact-byte mapping checks against the normal SGLang/FlashInfer loader helpers.

Run with CUDA and the serving image's FlashInfer. These checks are outside the
update timing path and deliberately use the normal value-layout implementation
as an independent oracle for the feature-owned byte-mask implementation.
test_expert_delta_matches_full_loader_layout_and_scale_refresh, the CuTe DSL
W4A16 expert-delta test, needs the companion SGLang tree (interleave_w13_halves).
"""

import math
import sys
from types import SimpleNamespace

import pytest
import torch
from tests.manual.gpu_delta.receiver_apply import apply_prepared_masks

from miles.backends.sglang_utils.gpu_delta.kernel_backends.flashinfer import FlashInferLayout, nvfp4_derived_update
from miles.utils.gpu_delta.publication import canonical_encoding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kind", ["ue8m0", "bf16-norm"])
def test_converted_small_xor_uses_prebound_kernel_and_is_exactly_reversible(kind):
    from unittest.mock import patch

    from miles.backends.sglang_utils.gpu_delta.bindings import TensorBinding, prepare_raw_xor
    from miles.backends.sglang_utils.gpu_delta.receiver import PreparedDelta
    from miles.utils.gpu_delta import apply

    device = torch.device("cuda", torch.cuda.current_device())
    if kind == "ue8m0":
        old = torch.arange(255, dtype=torch.uint8, device=device)
        new = old.flip(0)
        target = old.view(torch.float8_e8m0fnu).float().reshape(17, 15)
        expected = new.view(torch.float8_e8m0fnu).float().reshape(target.shape)
        binding = TensorBinding(
            "scales",
            "U8",
            (17, 15),
            [[0, 17], [0, 15]],
            (target,),
            encoding="xor_raw_bytes",
            format="fp8-block128-ue8m0",
        )
    else:
        bits = torch.arange(65536, dtype=torch.int32, device=device)
        nan = ((bits & 0x7F80) == 0x7F80) & ((bits & 0x007F) != 0)
        bits = bits[~nan]
        old = bits.to(torch.int16).view(torch.bfloat16)
        new = old.flip(0)
        target, expected = old.float(), new.float()
        # Cover every finite/subnormal value, both signed zeros and infinities.
        # Exclude NaN payloads whose cast canonicalization is implementation dependent.
        assert torch.equal(target.view(torch.int32), bits << 16)
        binding = TensorBinding(
            "norm", "BF16", (old.numel(),), [[0, old.numel()]], (target,), encoding="xor_raw_bytes"
        )
        old, new = old.view(torch.uint8), new.view(torch.uint8)
    original = target.clone()
    mask = old ^ new
    backend = SimpleNamespace(device=device, raw_plan=None)
    first_group = None
    for wanted in (expected, original):
        prepared = PreparedDelta(backend)
        prepared.error = torch.zeros(1, dtype=torch.int32, device=device)
        before = target.clone()
        updates = prepare_raw_xor([(binding, mask)])
        prepared._prepare_raw_apply(updates)
        assert torch.equal(target, before)
        if first_group is None:
            first_group = backend.raw_plan[1]
        else:
            assert backend.raw_plan[1] is first_group
        with patch.object(apply, "_compiled", side_effect=AssertionError("paused compilation")):
            prepared.raw_apply()
        assert torch.equal(target.view(torch.uint8), wanted.view(torch.uint8))
    if kind == "ue8m0":
        assert target.flatten().view(torch.int32)[0].item() == 0x00400000


def _reference_layer(values, independent_mma=False, tp_rank=0, tp_size=1):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        interleave_w13_halves,
        resolve_cutedsl_standard_scales,
    )
    from sglang.srt.layers.quantization.modelopt_quant import _compute_gemm1_alphas
    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.layers.utils import alias_or_bind_derived_param, copy_or_rebind_param

    layer = torch.nn.Module()
    layer.moe_tp_size, layer.moe_tp_rank = tp_size, tp_rank
    layer.use_padded_loading = False
    layer.use_presharded_weights = False
    layer.moe_ep_rank = 1
    layer.num_experts, layer.num_local_experts = 4, 2
    layer.quant_method = SimpleNamespace(_is_cutedsl_v2_standard=True)
    layer.moe_runner_config = SimpleNamespace(
        is_gated=True, activation="silu", gemm1_clamp_limit=None, swiglu_limit=None
    )
    layer.quant_config = SimpleNamespace(use_per_token_activation=False)
    layer._map_global_expert_id_to_local_expert_id = lambda expert: (expert - 2 if 2 <= expert < 4 else -1)
    layer._cutedsl_wrapper = SimpleNamespace(quant_mode="w4a16")
    for suffix in ("weight", "weight_scale"):
        gate, up, down = (values[projection, suffix] for projection in ("gate", "up", "down"))
        rows, cols = gate.shape[1] // tp_size, down.shape[2] // tp_size
        gate = gate[:, tp_rank * rows : (tp_rank + 1) * rows]
        up = up[:, tp_rank * rows : (tp_rank + 1) * rows]
        down = down[:, :, tp_rank * cols : (tp_rank + 1) * cols]
        # The actual checkpoint loader puts Up before Gate for CuTe DSL.
        fused = torch.cat([up, gate], dim=1)
        copy_or_rebind_param(layer, "w13_" + suffix, interleave_w13_halves(fused))
        copy_or_rebind_param(layer, "w2_" + suffix, down.clone())
    for stem in ("w13", "w2"):
        scale = getattr(layer, stem + "_weight_scale")
        alias_or_bind_derived_param(
            layer,
            stem + "_weight_scale",
            stem + "_blockscale_swizzled",
            swizzle_blockscale(scale),
        )
        weight = getattr(layer, stem + "_weight")
        mma = convert_sf_to_mma_layout(
            getattr(layer, stem + "_blockscale_swizzled").contiguous().view(torch.uint8).reshape(-1),
            m=weight.shape[1],
            k=2 * weight.shape[2],
            num_groups=2,
            sf_vec_size=16,
        )
        # Both public-loader aliasing and an independent existing MMA consumer
        # image must receive exactly one mask application.
        if independent_mma:
            mma = mma.clone(memory_format=torch.preserve_format)
        copy_or_rebind_param(layer, stem + "_blockscale_mma", mma)
    copy_or_rebind_param(
        layer,
        "w13_weight_scale_2",
        torch.stack([values["gate", "weight_scale_2"], values["up", "weight_scale_2"]], dim=1),
    )
    copy_or_rebind_param(layer, "w2_weight_scale_2", values["down", "weight_scale_2"].clone())
    # W4A16 ignores canonical activation scales for its GEMM alphas. Keep their
    # global checkpoint buffers distinct from the neutralized derived values.
    copy_or_rebind_param(layer, "w13_input_scale", torch.ones(4, 2, device="cuda"))
    copy_or_rebind_param(layer, "w2_input_scale", torch.ones(4, device="cuda"))
    neutral = torch.ones(2, device="cuda")
    gate, up = _compute_gemm1_alphas(layer.w13_weight_scale_2, neutral, True)
    copy_or_rebind_param(layer, "g1_alphas", gate)
    copy_or_rebind_param(layer, "g1_alphas_up", up)
    copy_or_rebind_param(layer, "g2_alphas", neutral * layer.w2_weight_scale_2)
    copy_or_rebind_param(layer, "w13_input_scale_quant", neutral.clone())
    copy_or_rebind_param(layer, "w2_input_scale_quant", neutral.clone())
    a, b, c, d = resolve_cutedsl_standard_scales(layer)
    layer._cutedsl_scales = tuple(x.clone() for x in (a, b, c))
    layer._cutedsl_input_scale = d.clone()
    return layer


def _canonical(seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    values = {}
    for projection, rows, cols in (
        ("gate", 128, 256),
        ("up", 128, 256),
        ("down", 256, 128),
    ):
        values[projection, "weight"] = torch.randint(
            256, (2, rows, cols // 2), dtype=torch.uint8, device="cuda", generator=rng
        )
        # Legal positive scale values; the resulting XOR masks are never
        # interpreted as FP8 numeric values by the delta mapper.
        values[projection, "weight_scale"] = torch.randint(
            1,
            120,
            (2, rows, cols // 16),
            dtype=torch.uint8,
            device="cuda",
            generator=rng,
        ).view(torch.float8_e4m3fn)
        values[projection, "weight_scale_2"] = torch.rand(2, device="cuda", generator=rng) + 0.5
    return values


@pytest.mark.parametrize("independent_mma", [False, True])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1)])
def test_expert_delta_matches_full_loader_layout_and_scale_refresh(independent_mma, tp_size, tp_rank):
    from miles.backends.sglang_utils.gpu_delta.bindings import _moe_binding, prepare_raw_xor
    from miles.backends.sglang_utils.gpu_delta.derived import plan_updates, prepare_updates
    from miles.backends.sglang_utils.gpu_delta.receiver import PreparedDelta

    old, new = _canonical(13), _canonical(29)
    live, expected = (
        _reference_layer(old, independent_mma, tp_rank, tp_size),
        _reference_layer(new, tp_rank=tp_rank, tp_size=tp_size),
    )
    images = {name: parameter for name, parameter in live.named_parameters(remove_duplicate=False)}
    images.update({f"wrapper.{i}": tensor for i, tensor in enumerate(live._cutedsl_scales)})
    pointers = {name: tensor.data_ptr() for name, tensor in images.items()}
    layout = FlashInferLayout("nvfp4", "cutedsl", True, "w4a16", up_first=True)
    bindings, masks, raw_items = [], [], []
    for (projection, suffix), before in old.items():
        after = new[projection, suffix]
        for local in range(2):
            value = before[local]
            name = f"model.layers.3.mlp.experts.{local + 2}.{projection}_proj.{suffix}"
            dtype = {
                torch.uint8: "U8",
                torch.float8_e4m3fn: "F8_E4M3",
                torch.float32: "F32",
            }[value.dtype]
            # The layout's inventory carries canonical_encoding's choice: raw masks for the scalar weight_scale_2.
            encoding = canonical_encoding(list(value.shape))
            binding = _moe_binding(
                name,
                {"dtype": dtype, "shape": list(value.shape), "encoding": encoding},
                live,
                local + 2,
                projection,
                suffix,
                layout,
            )
            mask = value.reshape(-1).view(torch.uint8) ^ after[local].reshape(-1).view(torch.uint8)
            if binding.encoding == "xor_raw_bytes":
                raw_items.append((binding, mask))
            else:
                bindings.append(binding)
                masks.append(mask)
    update = nvfp4_derived_update(live, layout)
    targets = [binding.storage[0] for binding, _ in raw_items]
    raw_updates = prepare_raw_xor(raw_items, {id(target) for target in targets})
    plans = plan_updates([update], targets)
    before = [target.clone() for target in update.destinations]
    derived = prepare_updates(
        plans, {id(target): predicted for target, _, predicted in raw_updates if predicted is not None}
    )
    for target, original in zip(update.destinations, before, strict=False):
        torch.testing.assert_close(target, original, rtol=0, atol=0)
    device = torch.device("cuda", torch.cuda.current_device())
    prepared = PreparedDelta(SimpleNamespace(device=device, raw_plan=None))
    prepared.error = torch.zeros(1, dtype=torch.int32, device=device)
    prepared._prepare_raw_apply(raw_updates)
    prepared.raw_apply()
    apply_prepared_masks(bindings, masks, derived)
    for name, target in expected.named_parameters(remove_duplicate=False):
        actual = dict(live.named_parameters(remove_duplicate=False))[name]
        torch.testing.assert_close(actual.view(torch.uint8), target.view(torch.uint8), rtol=0, atol=0)
    for actual, target in zip(live._cutedsl_scales, expected._cutedsl_scales, strict=False):
        torch.testing.assert_close(actual.view(torch.uint8), target.view(torch.uint8), rtol=0, atol=0)
    assert pointers == {name: tensor.data_ptr() for name, tensor in images.items()}
    assert live.w13_weight_scale.data_ptr() == live.w13_blockscale_swizzled.data_ptr()
    assert live.w2_weight_scale.data_ptr() == live.w2_blockscale_swizzled.data_ptr()


def test_scale_binding_matches_loader_padding():
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    from miles.backends.sglang_utils.gpu_delta.bindings import _moe_binding

    for shape in ((17, 3), (128, 64), (129, 8)):
        values = torch.randint(1, 120, shape, dtype=torch.uint8, device="cuda")
        physical = swizzle_blockscale(values.view(torch.float8_e4m3fn))
        live = physical.clone().unsqueeze(0)
        layer = SimpleNamespace(
            moe_tp_size=1,
            use_presharded_weights=False,
            quant_method=SimpleNamespace(_is_cutedsl_v2_standard=True),
            moe_runner_config=SimpleNamespace(is_gated=True),
            _map_global_expert_id_to_local_expert_id=lambda _: 0,
            w2_blockscale_swizzled=live,
            w2_weight_scale=live,
        )
        binding = _moe_binding(
            "model.layers.0.mlp.experts.0.down_proj.weight_scale",
            {"dtype": "F8_E4M3", "shape": list(shape)},
            layer,
            0,
            "down",
            "weight_scale",
            FlashInferLayout("nvfp4", "cutedsl", True, "w4a16", up_first=True),
        )
        mask = torch.randint(256, shape, dtype=torch.uint8, device="cuda")
        expected = swizzle_blockscale((values ^ mask).view(torch.float8_e4m3fn))
        apply_prepared_masks([binding], [mask.reshape(-1)])
        torch.testing.assert_close(live[0].view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def test_cached_mla_refresh_survives_graph_replay_and_failure_gate():
    from miles.backends.sglang_utils.gpu_delta.bindings import _direct_binding
    from miles.backends.sglang_utils.gpu_delta.models.deepseek_v2 import mla_derived_images

    weight = torch.zeros((2 * (4 + 6), 8), dtype=torch.bfloat16, device="cuda")
    key, value = weight.unflatten(0, (2, 10)).split([4, 6], dim=1)
    attn = SimpleNamespace(
        kv_b_proj=SimpleNamespace(weight=weight),
        qk_nope_head_dim=4,
        v_head_dim=6,
        w_kc=key.transpose(1, 2).contiguous().transpose(1, 2),
        w_vc=value.contiguous().transpose(1, 2),
    )
    derived = list(mla_derived_images("attention", attn))
    # TP4 rank3: local heads 6:8 of the canonical [8*(K+V), rank] kv_b_proj.
    binding = _direct_binding(
        "model.layers.0.self_attn.kv_b_proj.weight",
        {"dtype": "BF16", "shape": [80, 8], "encoding": "xor_bytes"},
        weight,
        [[60, 80], [0, 8]],
    )
    mask = torch.randint(256, (80 * 8 * 2,), dtype=torch.uint8, device="cuda")
    apply_prepared_masks([binding], [mask], derived)
    expected = mask.view(torch.bfloat16).reshape(8, 10, 8)
    torch.testing.assert_close(
        weight.view(torch.uint8),
        expected[6:].reshape(20, 8).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    _assert_refresh_survives_graph_replay(
        derived,
        lambda: [(image.destination, image.source) for image in derived],
        lambda: weight.view(torch.uint8).random_(256),
    )


def _assert_refresh_survives_graph_replay(images, pairs, change_sources):
    """Graph-captured refreshes publish fresh (destination, value) pairs on success and keep old bytes on failure."""
    identity = [(destination.data_ptr(), destination.stride()) for destination, _ in pairs()]
    succeeded = torch.tensor(True, device="cuda")

    def refresh():
        for image in images:
            image.refresh(succeeded)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        refresh()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        refresh()
    for success in (True, True, False):
        before = [destination.clone() for destination, _ in pairs()]
        change_sources()
        succeeded.fill_(success)
        graph.replay()
        torch.cuda.synchronize()
        for (destination, value), old in zip(pairs(), before, strict=True):
            expected = value if success else old
            assert torch.equal(destination.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    assert identity == [(destination.data_ptr(), destination.stride()) for destination, _ in pairs()]


def test_main_dcp_replicas_apply_once_through_the_kernel():
    """SGLang main's DCP storage through the prepared CUDA apply: aliased local q_b rows and transposed V."""
    from miles.backends.sglang_utils.gpu_delta.bindings import _direct_binding
    from miles.backends.sglang_utils.gpu_delta.models.deepseek_v2 import DcpReplica, with_dcp_replicas

    heads, nope, rope, value_dim, lora = 8, 128, 64, 128, 512
    old_q, new_q = (torch.randn(heads * (nope + rope), lora, device="cuda").bfloat16() for _ in range(2))
    old_kv, new_kv = (torch.randn(heads * (nope + value_dim), lora, device="cuda").bfloat16() for _ in range(2))
    # Attention TP4 / DCP2, rank 3: local heads 6:8; the DCP group gathers heads 4:8. As in SGLang main's
    # model_runner._prepare_replicated_q_proj, the local q_b rows are this rank's rows of the gathered image
    # and w_vc_qrep is all_gather(w_vc.contiguous()), i.e. [heads, kv_lora_rank, v_head_dim].
    q_rows, kv_rows = nope + rope, nope + value_dim
    q_image = old_q[4 * q_rows : 8 * q_rows].clone()
    q_local = q_image[2 * q_rows :]
    kv_local = old_kv[6 * kv_rows :].clone()
    gathered = old_kv.unflatten(0, (heads, kv_rows))[4:]
    w_kc_qrep = gathered[:, :nope].contiguous()
    w_vc_qrep = gathered[:, nope:].transpose(1, 2).contiguous()
    q = with_dcp_replicas(
        _direct_binding(
            "model.layers.0.self_attn.q_b_proj.weight",
            {"dtype": "BF16", "shape": list(old_q.shape), "encoding": "xor_bytes"},
            q_local,
            [[6 * q_rows, 8 * q_rows], [0, lora]],
        ),
        (DcpReplica(q_image, q_rows),),
    )
    kv = with_dcp_replicas(
        _direct_binding(
            "model.layers.0.self_attn.kv_b_proj.weight",
            {"dtype": "BF16", "shape": list(old_kv.shape), "encoding": "xor_bytes"},
            kv_local,
            [[6 * kv_rows, 8 * kv_rows], [0, lora]],
        ),
        (
            DcpReplica(w_kc_qrep, kv_rows, (0, nope)),
            DcpReplica(w_vc_qrep, kv_rows, (nope, kv_rows), transposed=True),
        ),
    )
    # The gathered image carries the aliased local rows: one segment, one XOR.
    assert q.storage == (q_image,) and len(q.segments) == 1
    masks = [
        (old.view(torch.uint8) ^ new.view(torch.uint8)).reshape(-1) for old, new in ((old_q, new_q), (old_kv, new_kv))
    ]
    apply_prepared_masks([q, kv], masks)
    new_heads = new_kv.unflatten(0, (heads, kv_rows))
    for live, expected in (
        (q_image, new_q[4 * q_rows : 8 * q_rows]),
        (q_local, new_q[6 * q_rows :]),
        (kv_local, new_kv[6 * kv_rows :]),
        (w_kc_qrep, new_heads[4:, :nope]),
        (w_vc_qrep, new_heads[4:, nope:].transpose(1, 2)),
    ):
        torch.testing.assert_close(
            live.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8), rtol=0, atol=0
        )


def test_canonical_shadow_first_fill_orders_its_allocation_stream_across_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from miles.backends.sglang_utils.gpu_delta.bindings import CanonicalShadows
    from miles.backends.sglang_utils.gpu_delta.layout import _side_stream

    name, meta = "model.norm.weight", {"dtype": "F32", "shape": [8]}
    expected = torch.arange(8, dtype=torch.float32)
    checkpoint = SimpleNamespace(
        inventory={name: meta}, read_rows=lambda name, start, stop: expected[start:stop].view(torch.uint8).numpy()
    )
    target = torch.zeros(8, device="cuda")
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    store = {}
    shadows = CanonicalShadows(store, create=True)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(200_000_000)
        target.fill_(-1)
        # Model an allocator returning storage with a pending write on its allocation stream.
        # The returned tensor stays alive; this isolates ordering from allocator cache heuristics.
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *args, **kwargs: target)
            shadow = shadows.get(name, meta, (0, 8), target.device)

    def fill():
        with torch.cuda.device(target.device), _side_stream((shadow,)):
            shadows.fill(checkpoint, (name,))

    # Preparation's thread has a different current stream from describe's producer.
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(fill).result()
    producer.synchronize()
    torch.testing.assert_close(shadow.cpu(), expected, rtol=0, atol=0)

    # Re-admission retains committed bytes and never refills or waits on the old allocation.
    shadow.add_(10)
    retained = CanonicalShadows(store, create=False)
    assert retained.get(name, meta, (0, 8), target.device) is shadow
    retained.fill(checkpoint, (name,))
    torch.testing.assert_close(shadow.cpu(), expected + 10, rtol=0, atol=0)


def test_computed_images_survive_graph_replay_and_failure_gate():
    from miles.backends.sglang_utils.gpu_delta import layout
    from miles.backends.sglang_utils.gpu_delta.bindings import ComputedImage, ConsumerSnapshot, _direct_binding

    # An FP32 A = -exp(A_log) of a BF16 tensor, and Kimi-K3's F32 o_norm served as BF16 plus an FP32 decode copy.
    a_log = torch.randn(8, device="cuda").bfloat16()
    a = -torch.exp(a_log.float())
    norm = torch.randn(16, device="cuda")
    norm_bf16 = norm.to(torch.bfloat16)
    norm_fp32 = norm_bf16.float()
    images = [
        ComputedImage("mixer.A", (a_log,), (a,), lambda log: (-torch.exp(log.float()),)),
        ComputedImage(
            "o_norm",
            (norm,),
            (norm_bf16, norm_fp32),
            lambda weight: (weight.to(torch.bfloat16), weight.to(torch.bfloat16).float()),
        ),
    ]
    # Admission verifies on a side stream that each image reproduces its consumer bytewise.
    plan = layout.GpuDeltaLayout.__new__(layout.GpuDeltaLayout)
    plan.bindings = [
        _direct_binding("mixer.A_log", {"dtype": "BF16", "shape": [8], "encoding": "xor_raw_bytes"}, a_log),
        _direct_binding("o_norm.weight", {"dtype": "F32", "shape": [16], "encoding": "xor_raw_bytes"}, norm),
    ]
    plan.derived, plan.computed = [], images
    # The mapping pins the served consumers while binding; computed images write only pinned tensors.
    plan._parameter_roots, plan._consumers = [], [ConsumerSnapshot(lambda: (a, norm_bf16, norm_fp32))]
    a.zero_()
    torch.cuda.current_stream().synchronize()
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(200_000_000)
        a.copy_(-torch.exp(a_log.float()))
        # Admission's side stream must observe images just produced by mapping.finish().
        plan._admit_images([], {})
    a.view(torch.int32)[3] ^= 1
    with pytest.raises(ValueError, match="computed image mixer.A does not reproduce its live consumer"):
        plan._admit_images([], {})
    a.copy_(-torch.exp(a_log.float()))

    def change_sources():
        a_log.copy_(torch.randn(8, device="cuda"))
        norm.copy_(torch.randn(16, device="cuda"))

    def pairs():
        return [pair for image in images for pair in zip(image.destinations, image.values(), strict=True)]

    _assert_refresh_survives_graph_replay(images, pairs, change_sources)


def _matrix_canonical(format, seed, hidden, intermediate):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    values = {}
    for projection, rows, cols in (
        ("gate", intermediate, hidden),
        ("up", intermediate, hidden),
        ("down", hidden, intermediate),
    ):
        packed = format == "nvfp4"
        shape = (2, rows, cols // (2 if packed else 1))
        if format == "bf16":
            weight = torch.randn(shape, device="cuda", generator=rng).bfloat16()
        else:
            weight = torch.randint(
                1,
                256 if packed else 120,
                shape,
                dtype=torch.uint8,
                device="cuda",
                generator=rng,
            )
            if not packed:
                weight = weight.view(torch.float8_e4m3fn)
        values[projection, "weight"] = weight
        if format == "bf16":
            continue
        block = 16 if format == "nvfp4" else 32
        scale_rows = rows
        if format == "fp8-block128-ue8m0":
            block, scale_rows = 128, math.ceil(rows / 128)
        scale = torch.randint(
            1 if format == "nvfp4" else 118,
            120 if format == "nvfp4" else 136,
            (2, scale_rows, math.ceil(cols / block)),
            dtype=torch.uint8,
            device="cuda",
            generator=rng,
        )
        if format == "nvfp4":
            scale = scale.view(torch.float8_e4m3fn)
        elif format == "fp8-block128-ue8m0":
            # Include the UE8M0 subnormal and largest finite scale endpoints.
            scale.flatten()[0], scale.flatten()[-1] = 0, 254
        values[projection, "scale"] = scale
    return values


def _matrix_loader_reference(values, layout, hidden, intermediate):
    """Run ordinary preprocessing, independent of GPU-delta address formulas."""
    layer = torch.nn.Module()
    layer.moe_tp_size = 1
    layer.quant_method = SimpleNamespace()
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.num_local_experts = 2
    layer.intermediate_size_per_partition = intermediate
    layer._map_global_expert_id_to_local_expert_id = lambda expert: expert - 2
    scale_name = "weight_scale" if layout.format == "nvfp4" else "weight_scale_inv"

    def bind(name, value):
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))

    for suffix in ("weight",) if layout.format == "bf16" else ("weight", "scale"):
        name = "weight" if suffix == "weight" else scale_name
        bind("w13_" + name, torch.cat([values["gate", suffix], values["up", suffix]], 1))
        bind("w2_" + name, values["down", suffix].clone())
    if layout.backend == "megamoe":
        from flashinfer.moe_ep import (
            MoEWeightPack,
            preprocess_mxfp8_cutedsl_mega_weights,
            preprocess_nvfp4_cutedsl_mega_weights,
        )

        preprocess = {
            "nvfp4": preprocess_nvfp4_cutedsl_mega_weights,
            "mxfp8": preprocess_mxfp8_cutedsl_mega_weights,
        }[layout.format]
        if layout.format == "nvfp4" and layout.quant_mode == "w4a16":
            from flashinfer.moe_ep import preprocess_bf16_nvfp4_cutedsl_mega_weights

            preprocess = preprocess_bf16_nvfp4_cutedsl_mega_weights
        weights = MoEWeightPack(
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            w13_scale=getattr(layer, "w13_" + scale_name),
            w2_scale=getattr(layer, "w2_" + scale_name),
        )
        transformed = preprocess(weights, intermediate_size=intermediate, hidden_size=hidden)
        for stem, (weight, scale) in zip(("w13", "w2"), transformed, strict=False):
            bind(stem + "_weight", weight)
            bind(stem + "_" + scale_name, scale)
    elif layout.format == "bf16":
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            convert_to_block_layout,
            get_w2_permute_indices_with_cache,
        )

        cache = {}
        for stem, permutation in (
            ("w13", _maybe_get_cached_w3_w1_permute_indices),
            ("w2", get_w2_permute_indices_with_cache),
        ):
            images = []
            for weight in getattr(layer, stem + "_weight"):
                raw = weight.view(torch.uint8)
                indices = permutation(cache, raw, 128).to(raw.device)
                images.append(convert_to_block_layout(raw[indices].contiguous(), 128).view(torch.bfloat16))
            bind(stem + "_weight", torch.stack(images))
    elif layout.format == "nvfp4":
        from sglang.srt.layers.quantization.utils import prepare_static_weights_for_trtllm_fp4_moe

        # The current ModelOpt loader stores TRTLLM weights/scales [Up, Gate].
        w13_input = torch.cat([values["up", "weight"], values["gate", "weight"]], 1)
        s13_input = torch.cat([values["up", "scale"], values["gate", "scale"]], 1)
        w13, s13, w2, s2 = prepare_static_weights_for_trtllm_fp4_moe(
            w13_input,
            layer.w2_weight,
            s13_input,
            layer.w2_weight_scale,
            hidden,
            intermediate,
            2,
        )
        for name, value in zip(
            ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
            (w13, s13, w2, s2),
            strict=False,
        ):
            bind(name, value)
    elif layout.format == "mxfp8":
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import align_mxfp8_moe_weights_for_flashinfer_trtllm

        align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)
    elif layout.format == "fp8-block128-ue8m0":
        for stem in ("w13", "w2"):
            scale = getattr(layer, stem + "_weight_scale_inv")
            # Numerical powers of two independently check the byte-to-FP32 API conversion.
            bind(
                stem + "_weight_scale_inv",
                torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.int() - 127),
            )
    else:
        raise ValueError(f"no native preprocessing reference for {layout}")
    return layer


@pytest.mark.parametrize(
    "format,backend,intermediate,quant_mode",
    [
        ("bf16", "trtllm", 256, ""),
        ("fp8-block128-ue8m0", "trtllm", 256, ""),
        ("nvfp4", "trtllm", 256, "w4a4"),
        ("mxfp8", "trtllm", 256, ""),
        ("nvfp4", "megamoe", 320, "w4a16"),
        ("mxfp8", "megamoe", 256, ""),
    ],
)
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1)])
def test_flashinfer_matrix_matches_native_preprocessing(format, backend, intermediate, quant_mode, tp_size, tp_rank):
    from miles.backends.sglang_utils.gpu_delta.bindings import _moe_binding, prepare_raw_xor

    hidden = 384
    layout = FlashInferLayout(
        format,
        backend,
        True,
        quant_mode,
        up_first=format == "nvfp4" and backend == "trtllm",
    )
    old, new = (_matrix_canonical(format, seed, hidden, intermediate * tp_size) for seed in (17, 43))
    local_values = []
    for values in (old, new):
        local = {}
        for (projection, suffix), value in values.items():
            axis = 2 if projection == "down" else 1
            size = value.shape[axis] // tp_size
            local[projection, suffix] = value.narrow(axis, tp_rank * size, size)
        local_values.append(local)
    live, expected = (_matrix_loader_reference(values, layout, hidden, intermediate) for values in local_values)
    live.moe_tp_size, live.moe_tp_rank = tp_size, tp_rank
    live.use_padded_loading = backend == "trtllm"
    images = dict(live.named_parameters())
    identity = {name: (value.data_ptr(), value.stride()) for name, value in images.items()}
    before = {name: value.contiguous().view(torch.uint8).clone() for name, value in images.items()}
    bindings, masks, complete = [], [], []
    for (projection, kind), original in old.items():
        suffix = kind
        if kind == "scale":
            suffix = "weight_scale" if format == "nvfp4" else "weight_scale_inv"
        dtype = {
            torch.bfloat16: "BF16",
            torch.uint8: "U8",
            torch.float8_e4m3fn: "F8_E4M3",
        }[original.dtype]
        for expert in range(2):
            value, updated = original[expert], new[projection, kind][expert]
            binding = _moe_binding(
                f"model.layers.3.mlp.experts.{expert + 2}.{projection}_proj.{suffix}",
                {"dtype": dtype, "shape": list(value.shape)},
                live,
                expert + 2,
                projection,
                suffix,
                layout,
            )
            payload = updated.contiguous().view(torch.uint8)
            if binding.encoding == "xor_raw_bytes":
                mask = value.contiguous().view(torch.uint8) ^ payload
                complete.extend(prepare_raw_xor([(binding, mask)]))
            else:
                bindings.append(binding)
                masks.append((value.contiguous().view(torch.uint8) ^ payload).flatten())
    # Admission and complete-value preparation must not modify live parameters.
    for name, value in images.items():
        torch.testing.assert_close(value.contiguous().view(torch.uint8), before[name], rtol=0, atol=0)
    if complete:
        for target, mask, _ in complete:
            target.view(torch.uint8).bitwise_xor_(mask.view(target.view(torch.uint8).shape))
    apply_prepared_masks(bindings, masks)
    for name, value in expected.named_parameters():
        torch.testing.assert_close(
            images[name].contiguous().view(torch.uint8),
            value.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
    assert identity == {name: (value.data_ptr(), value.stride()) for name, value in images.items()}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
