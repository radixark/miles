"""Host wiring of the deterministic chunked GDN kernels (``miles/kernels/attention/gdn/cake``).

One exact chunk table per micro-batch layout (shared by every layer, no device-to-host synchronisation), launch
counts, and the bit-identical tape / recompute backward.  Needs a CUDA device; the kernel tests need Blackwell
(SM100a / SM103a) and skip elsewhere.
"""

import itertools
import math
import warnings

import pytest

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1200, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles.kernels.attention.gdn.cake import ops  # noqa: E402

# A real Qwen3.8-27B TP4 micro-batch: three sequences plus the 512-token alignment pad as its own segment.
LAYOUT = (3339, 2802, 1340, 263)
CHUNK = ops.CHUNK
DEVICE = torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(autouse=True)
def _no_ambient_segment_override(monkeypatch):
    # a leaked override would turn the policy arms into the serial path with every assertion still passing
    monkeypatch.delenv("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS", raising=False)
    monkeypatch.delenv("MILES_GDN_CHUNK_TRAIN_SCAN_CFG", raising=False)
    monkeypatch.delenv("MILES_GDN_CHUNK_TRAIN_FINALIZE", raising=False)  # R7: the finalize kernel pin
    monkeypatch.setattr(ops, "FORCED_FINALIZE", None)
    ops._scan_plans_memo.cache_clear()  # the plan memo (R4) starts empty in every test
    ops.host_cache_clear()  # the R6-7 host caches (workspace pool, chunk-table front cache, device arch) start empty too


def _cu(seq_lens):
    return (0, *itertools.accumulate(seq_lens))


def _expected_table(seq_lens):
    starts, lens, seq_chunk_start, token = [], [], [0], 0
    for n in seq_lens:
        for c in range(0, n, CHUNK):
            starts.append(token + c)
            lens.append(min(CHUNK, n - c))
        seq_chunk_start.append(len(starts))
        token += n
    return starts, lens, seq_chunk_start


def test_host_chunk_table_is_built_once_per_layout():
    ops._host_chunk_meta.cache_clear()
    before = ops._HOST_META_BUILDS
    metas = [
        ops.chunk_meta_from_host(_cu(LAYOUT), DEVICE, 4) for _ in range(48)
    ]  # R3 C6: the head count is part of the plan key
    assert ops._HOST_META_BUILDS == before + 1, "48 layers of one micro-batch must share one chunk table"
    info = ops._scan_plans_memo.cache_info()  # R4: the joint scan plan is searched once per layout, not per layer
    assert (info.hits, info.misses) == (47, 1), info
    assert all(m is metas[0] for m in metas)
    starts, lens, seq_chunk_start = _expected_table(LAYOUT)
    assert metas[0].num_chunks == len(starts) and metas[0].num_seqs == len(LAYOUT)
    assert metas[0].chunk_start.tolist() == starts
    assert metas[0].chunk_len.tolist() == lens
    assert metas[0].seq_chunk_start.tolist() == seq_chunk_start
    assert all(t.dtype == torch.int32 and t.is_cuda for t in (metas[0].chunk_start, metas[0].chunk_len))
    # the same layout from a CPU tensor hits the cache; another layout is another build
    assert ops.chunk_meta_from_host(torch.tensor(_cu(LAYOUT), dtype=torch.int64), DEVICE, 4) is metas[0]
    other = ops.chunk_meta_from_host(_cu((8192, 512)), DEVICE, 4)
    assert other is not metas[0] and ops._HOST_META_BUILDS == before + 2


def test_host_boundaries_are_validated():
    with pytest.raises(ValueError):
        ops.chunk_meta_from_host((0, 10, 5), DEVICE, 4)
    with pytest.raises(ValueError):
        ops.chunk_meta_from_host((4,), DEVICE, 4)
    with pytest.raises(ValueError):
        ops.chunk_meta_from_host(torch.tensor([0, 64], device=DEVICE), DEVICE, 4)


def _require_blackwell():
    from miles.kernels.attention.gdn.cake import SUPPORTED_CAPABILITIES

    if torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES:
        pytest.skip("the deterministic GDN kernels need SM100a / SM103a")
    pytest.importorskip("fla.ops.gated_delta_rule")


def _inputs(seq_lens, num_k_heads=2, num_v_heads=4, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(seq_lens)
    q = torch.randn(1, total, num_k_heads, 128, device="cuda", generator=gen).to(torch.bfloat16)
    k = torch.randn(1, total, num_k_heads, 128, device="cuda", generator=gen).to(torch.bfloat16)
    v = torch.randn(1, total, num_v_heads, 128, device="cuda", generator=gen).to(torch.bfloat16)
    g = -torch.rand(1, total, num_v_heads, device="cuda", generator=gen) * 0.5
    beta = torch.rand(1, total, num_v_heads, device="cuda", generator=gen).to(torch.bfloat16)
    do = torch.randn(1, total, num_v_heads, 128, device="cuda", generator=gen).to(torch.bfloat16)
    return q, k, v, g, beta, do


def _leaves(inputs):
    q, k, v, g, beta, _do = inputs
    return [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]


def _step(inputs, leaves=None, **kwargs):
    do = inputs[-1]
    leaves = _leaves(inputs) if leaves is None else leaves
    for t in leaves:
        t.grad = None
    out, _ = ops.chunk_gated_delta_rule(*leaves, use_qk_l2norm_in_kernel=True, **kwargs)
    out.backward(do)
    return {"out": out.detach(), **{n: t.grad for n, t in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves, strict=True)}}


def test_host_table_matches_the_device_table():
    _require_blackwell()
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    arch = ops.device_arch(cu.device)
    plans = ops.scan_plans(sum(LAYOUT), len(LAYOUT), 4, DEVICE.index, arch)
    dev = ops.build_chunk_meta_from_cu_seqlens(cu, sum(LAYOUT), arch, scan_plans=plans)
    host = ops.chunk_meta_from_host(_cu(LAYOUT), cu.device, 4, arch=arch)
    n = host.num_chunks
    assert dev.num_chunks >= n
    assert torch.equal(dev.chunk_start[:n], host.chunk_start)
    assert torch.equal(dev.chunk_len[:n], host.chunk_len)
    assert torch.equal(dev.seq_chunk_start, host.seq_chunk_start)
    assert bool((dev.chunk_len[n:] == 0).all())


def test_host_and_device_tables_give_identical_results(monkeypatch):
    # The device route prices the chunk bound, the host route the exact layout, so the two routes may pick different
    # segment lengths (design doc: "bitwise when the segment lengths agree, within tolerance otherwise").  Like with
    # like: at one pinned segment length the chunk tables and every result are bit-identical across the routes; at the
    # planner's own picks the results agree within the BF16 contract tolerance.
    _require_blackwell()
    arch = ops.device_arch(DEVICE)
    inputs = _inputs(LAYOUT)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    host_fwd, _host_bwd = ops.segment_lens(sum(LAYOUT), len(LAYOUT), 4, DEVICE.index, arch, LAYOUT)
    assert host_fwd > 0, "LAYOUT must split on the host route for the pinned comparison to exercise a segment table"
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", host_fwd)
    a = _step(inputs, cu_seqlens=cu)
    b = _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    for name in a:
        assert torch.equal(
            a[name], b[name]
        ), f"{name} differs between the device and host chunk tables at L={host_fwd}"
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", None)
    a = _step(inputs, cu_seqlens=cu)
    b = _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    for name in a:
        assert torch.allclose(
            a[name].float(), b[name].float(), atol=1e-2, rtol=1e-2
        ), f"{name} differs between the routes' own picks"


def test_tape_and_recompute_backward_are_bit_identical(monkeypatch):
    _require_blackwell()
    inputs = _inputs(LAYOUT, seed=1)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    results = {}
    for save_state in (True, False):
        monkeypatch.setattr(ops, "SAVE_STATE", save_state)
        results[save_state] = _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    for name in results[True]:
        assert torch.equal(results[True][name], results[False][name]), f"{name}: tape vs recompute differ"


def test_wiring_has_no_host_sync_and_fixed_launches_per_layer(monkeypatch):
    _require_blackwell()
    from torch.profiler import ProfilerActivity, profile

    monkeypatch.setattr(ops, "SAVE_STATE", True)
    inputs = _inputs(LAYOUT, seed=2)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    kwargs = dict(cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    leaves = _leaves(inputs)  # the autograd leaves are created outside the profiled region (clone = a device copy)
    _step(inputs, leaves, **kwargs)  # warm-up: kernels loaded, chunk table built (one pinned host-to-device copy)
    torch.cuda.synchronize()
    builds = ops._HOST_META_BUILDS
    layers = 3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(layers):
            _step(inputs, leaves, **kwargs)
        torch.cuda.synchronize()
    assert ops._HOST_META_BUILDS == builds, "the chunk table must be reused across layers"
    runtime = [e.name for e in prof.events() if e.name.startswith(("cuda", "cu")) and not e.name.startswith("cudnn")]
    launches = [n for n in runtime if "LaunchKernel" in n]
    syncs = [n for n in runtime if "Synchronize" in n and n != "cudaDeviceSynchronize"]
    copies = [n for n in runtime if "Memcpy" in n]
    meta = ops.chunk_meta_from_host(_cu(LAYOUT), DEVICE, 4)
    split = [s.segment_len > 0 and s.num_segments > meta.num_seqs for s in (meta.fwd_segments, meta.bwd_segments)]
    # 4 forward + 5 backward kernels, plus the pre-scan of a segmented forward scan (its stitch is folded into the split
    # scan's prologue) and pre-scan + stitch of a segmented backward scan
    per_layer = 9 + int(split[0]) + 2 * int(split[1])
    assert len(launches) == layers * per_layer, (len(launches), per_layer, sorted(set(runtime)))
    assert not syncs and not copies, sorted(set(runtime))


def test_forward_launches_the_split_scan_on_split_plans_and_the_plain_scan_on_serial_plans(monkeypatch):
    """A split forward plan launches ``fwd_h_pre_*`` then ``fwd_h_r*_split`` and no forward stitch; a serial plan launches
    the plain ``fwd_h_r*`` alone.  The backward keeps its stitch on split plans (``reverse = 1``)."""
    _require_blackwell()
    arch = ops.device_arch(DEVICE)
    for vb, ks in ops.REG_SCAN_CONFIGS:
        stage = f"fwd_h_r{vb}k{ks}"
        assert ops._split_scan_stage(stage, arch) == f"{stage}_split"
        base, split = ops.MODULES[stage][arch], ops.MODULES[f"{stage}_split"][arch]
        assert base["kernel_symbol"] != split["kernel_symbol"] and base["block"] == split["block"]
        assert [tuple(a) for a in split["arg_plan"]][:-4] == [tuple(a) for a in base["arg_plan"]][:-3]
        assert tuple(split["arg_plan"][-4]) == ("buffer", "seg_flag")
    assert ops._split_scan_stage("dhu_r32k2", arch) is None
    launched = []
    real_launcher = ops.launcher

    def spy(stage, arch_, names):
        launched.append(stage)
        return real_launcher(stage, arch_, names)

    monkeypatch.setattr(ops, "launcher", spy)
    inputs = _inputs(LAYOUT, seed=6)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    results = {}
    for forced in (0, 4):  # serial scans, forced four-chunk segments
        monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", forced)
        ops._host_chunk_meta.cache_clear()
        launched.clear()
        results[forced] = _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
        fwd_scans = [s for s in launched if s.startswith("fwd_h_")]
        stitches = launched.count("stitch")
        if forced:
            assert (
                len(fwd_scans) == 2 and fwd_scans[0].startswith("fwd_h_pre_r") and fwd_scans[1].endswith("_split")
            ), fwd_scans
            assert fwd_scans[1] == fwd_scans[0].replace("fwd_h_pre_r", "fwd_h_r") + "_split", fwd_scans
            assert stitches == 1, launched  # the backward's stitch only
        else:
            assert (
                len(fwd_scans) == 1 and fwd_scans[0].startswith("fwd_h_r") and not fwd_scans[0].endswith("_split")
            ), fwd_scans
            assert stitches == 0, launched
    _allclose(results[4], results[0])


# --------------------------------------------------------------------------- #
# segment-parallel state scans (phase 2, task 1, round R2)
# --------------------------------------------------------------------------- #
def _allclose(a, b):
    for name in a:
        atol = rtol = 1e-3 if name == "dg" else 1e-2
        assert torch.allclose(a[name].float(), b[name].float(), atol=atol, rtol=rtol), name


def test_segment_tables_host_match_device(monkeypatch):
    _require_blackwell()
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", 3)
    ops._host_chunk_meta.cache_clear()
    arch = ops.device_arch(DEVICE)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    host = ops.chunk_meta_from_host(_cu(LAYOUT), DEVICE, 4, arch=arch)
    plans = ops.scan_plans(
        sum(LAYOUT), len(LAYOUT), 4, DEVICE.index, arch
    )  # the pinned length, planned configurations
    dev = ops.build_chunk_meta_from_cu_seqlens(cu, sum(LAYOUT), arch, scan_plans=plans)
    seq_chunk_start = _expected_table(LAYOUT)[2]
    for kind in ("fwd_segments", "bwd_segments"):
        h, d = getattr(host, kind), getattr(dev, kind)
        assert h.segment_len == d.segment_len == 3 and h.scan_config == d.scan_config == (32, 2)
        n = h.num_segments
        assert d.num_segments >= n
        assert torch.equal(d.seg_chunk_start[: n + 1], h.seg_chunk_start)
        assert torch.equal(d.seg_seq[:n], h.seg_seq)
        assert torch.equal(d.seq_seg_start, h.seq_seg_start)
        # the device table pads to its upper bound with empty segments that no kernel touches
        assert bool((d.seg_seq[n:] == -1).all()) and bool((d.seg_chunk_start[n + 1 :] == host.num_chunks).all())
        # every sequence's chunks are covered exactly once, in order, by segments of one to three chunks
        starts, seqs, sstart = h.seg_chunk_start.tolist(), h.seg_seq.tolist(), h.seq_seg_start.tolist()
        for i in range(len(LAYOUT)):
            segs = range(sstart[i], sstart[i + 1])
            assert starts[segs.start] == seq_chunk_start[i] and starts[segs.stop] == seq_chunk_start[i + 1]
            assert all(seqs[s] == i and 1 <= starts[s + 1] - starts[s] <= 3 for s in segs)


def test_segmented_scans_match_the_serial_scans(monkeypatch):
    _require_blackwell()
    inputs = _inputs(LAYOUT, seed=3)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    kwargs = dict(cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    results = {}
    for forced in (0, 4, None):  # serial scans, forced four-chunk segments, the production policy
        monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", forced)
        ops._host_chunk_meta.cache_clear()
        results[forced] = _step(inputs, **kwargs)
    _allclose(results[4], results[0])
    _allclose(results[None], results[0])
    # the device-table path of the segmented scans is bit-identical to the host-table path
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", 4)
    ops._host_chunk_meta.cache_clear()
    dev = _step(inputs, cu_seqlens=cu)
    for name in dev:
        assert torch.equal(dev[name], results[4][name]), f"{name} differs between the device and host segment tables"


def test_segmented_scans_are_bit_deterministic(monkeypatch):
    _require_blackwell()
    inputs = _inputs(LAYOUT, seed=4)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", 4)
    ops._host_chunk_meta.cache_clear()
    first = _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    for _ in range(5):
        again = _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
        for name in first:
            assert torch.equal(first[name], again[name]), f"{name}: repeated segmented passes differ"


# --------------------------------------------------------------------------- #
# segment planner (round R2b): the calibrated cost model is a pure function shared with the generator's
# ``gdn_chunk_training.plan_segments`` (same constants, same table below), so both tables split identically.
# --------------------------------------------------------------------------- #
B200_SMS, GB300_SMS = 148, 152  # SM counts the gates run on (the GB300 node reports 152 SMs; the full die has 160)
HV12_CTAS = 12 * (ops.HEAD_DIM // 32)  # CTAs per segment of the (32,2) scan at TP4 (12 value heads)
HV48_CTAS = 48 * (ops.HEAD_DIM // 32)  # TP1


def test_wave_factor_prices_co_residency_and_waves():
    assert ops._wave_factor(148, 148, 2) == 1.0 and ops._wave_factor(1, 148, 1) == 1.0  # one wave, one CTA per SM
    # a second co-resident CTA per SM shares the SM's issue slots
    assert ops._wave_factor(149, 148, 2) == ops.SCAN_CO_RESIDENT_FACTOR
    assert ops._wave_factor(296, 148, 2) == ops.SCAN_CO_RESIDENT_FACTOR
    # beyond the resident set: further waves; a tail wave behind a full wave starts staggered and shares SMs with it,
    # so it runs at the full wave's factor (R3 re-fit), not at 1.0
    assert ops._wave_factor(297, 148, 2) == 2 * ops.SCAN_CO_RESIDENT_FACTOR
    assert ops._wave_factor(149, 148, 1) == 2.0 and ops._wave_factor(297, 148, 1) == 3.0
    assert ops._wave_factor(0, 148, 2) == 0.0


def test_scan_makespan_prices_dispatch_order_co_residency_and_slot_release():
    # one sequence: steps x the wave factor of the whole grid; the real 8481 + 223 micro-batch: the pad's 48 CTAs make
    # the three-segment grid 192 CTAs, two per SM on 32 SMs for the WHOLE scan (dispatch order; measured 1.53x), six
    # segments (336 CTAs) start the 16 CTAs beyond the resident set when the pad finishes (measured 1.85x over 23);
    # at one CTA per SM there is no factor and the 32 extra CTAs start after the pad's 4 steps
    assert ops._scan_makespan((129,), (3,), HV12_CTAS, GB300_SMS, 2) == 43.0
    assert ops._scan_makespan((133, 4), (3, 1), HV12_CTAS, GB300_SMS, 2) == pytest.approx(
        45 * ops.SCAN_CO_RESIDENT_FACTOR
    )
    assert ops._scan_makespan((133, 4), (6, 1), HV12_CTAS, GB300_SMS, 2) == pytest.approx(
        (4 + 23) * ops.SCAN_CO_RESIDENT_FACTOR
    )
    assert ops._scan_makespan((133, 4), (3, 1), HV12_CTAS, GB300_SMS, 1) == 49.0
    assert ops._scan_makespan((38, 38, 38, 15, 3), (1,) * 5, HV12_CTAS, GB300_SMS, 2) == pytest.approx(
        38 * ops.SCAN_CO_RESIDENT_FACTOR
    )


def test_planner_constants_mirror_generator_verbatim():
    # the same literals as the generator's gdn_chunk_training.py (its test_gdn_chunk_segment_planner.py pins them)
    assert (ops.MIN_SEGMENT_CHUNKS, ops.SCAN_CO_RESIDENT_FACTOR, ops.PRESCAN_STEP_FACTOR) == (4, 1.60, 1.16)
    assert (ops.PRESCAN_WARMUP_CHUNKS, ops.STITCH_FIXED_CHUNKS, ops.STITCH_COPY_CHUNKS) == (3.3, 2.75, 2.3)
    assert (ops.SPLIT_GAIN_THRESHOLD, ops.MAX_SCAN_ROUNDS, ops.MAX_SCAN_WAVES_SERIAL) == (0.95, 2, 4)
    # both split bounds are four CTAs per SM: two rounds of two co-resident CTAs, or four sequential waves (R6)
    assert ops.split_cta_limit(148, 2) == ops.split_cta_limit(148, 1) == 592 and ops.split_cta_limit(152, 1) == 608
    assert ops.WARMUP_THRESH_LOG2 == -10.0 / math.log(2.0)
    # R3 (C6): per-configuration step factors and the configurations with pre-scan stages
    assert ops.SCAN_STEP_FACTOR == {(32, 2): 1.0, (32, 1): 1.18, (64, 1): 1.27}
    assert ops.PRESCAN_CONFIGS == ((32, 2), (64, 1))
    # R7: the one-wave finalize's group and launch-bound residency (generator: FINALIZE_MLP_GROUP / FINALIZE_MLP_MIN_BLOCKS)
    assert (ops.FINALIZE_MLP_GROUP, ops.FINALIZE_MLP_MIN_BLOCKS) == (3, 4)
    assert (
        ops.FINALIZE_MLP_MIN_BLOCKS * 128 * 128 <= 65536
    )  # four 128-thread CTAs at the 128-register cap fit the register file


# --------------------------------------------------------------------------- #
# finalize kernel selection (round R7): the one-wave kernel for value-head group 3 on a grid co-resident at four CTAs per SM
# --------------------------------------------------------------------------- #
def _chunks(seq_lens):
    return sum(-(-n // CHUNK) for n in seq_lens)


@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "num_v_heads", "sm_count", "expected"),
    [
        ((8192,), 4, 12, 148, "finalize_mlp"),  # 1x8192 TP4 on a 148-SM device: 512 CTAs <= 592
        ((8192,), 4, 12, 160, "finalize_mlp"),  # 160-SM device: 512 <= 640
        (LAYOUT, 4, 12, 148, "finalize_mlp"),  # the real TP4 micro-batch: 123 chunks x 4 = 492
        ((2384, 2382, 2369, 902, 155), 4, 12, 148, "finalize_mlp"),  # five-sequence 8192-token micro-batch: 528
        ((16384,), 4, 12, 160, "finalize"),  # 1024 > 640
        ((4096,) * 4, 4, 12, 160, "finalize"),  # 1024 > 640
        ((8192,), 8, 24, 160, "finalize"),  # TP2: 1024 > 640
        ((8192,), 16, 48, 160, "finalize"),  # TP1: 2048 > 640
        (LAYOUT, 2, 4, 148, "finalize"),  # value-head group 2 never selects the unrolled kernel
        ((64, 130, 5), 2, 4, 148, "finalize"),
        ((96, 40), 2, 6, 148, "finalize_mlp"),  # group 3 on a 6-CTA grid
    ],
)
def test_finalize_stage_mirrors_generator(seq_lens, num_heads, num_v_heads, sm_count, expected):
    assert ops.finalize_stage(_chunks(seq_lens), num_heads, num_v_heads, sm_count) == expected
    assert expected in ops.LAUNCH_ARGS and ops.LAUNCH_ARGS[expected] == ops._FINALIZE_ARGS


def test_finalize_stage_flips_at_the_launch_bound_wave_and_validates_the_contract(monkeypatch):
    for sm_count in (132, 148, 160):
        assert ops.finalize_stage(sm_count, 4, 12, sm_count) == "finalize_mlp"
        assert ops.finalize_stage(sm_count + 1, 4, 12, sm_count) == "finalize"
    with pytest.raises(ValueError):
        ops.finalize_stage(4, 3, 4, 148)
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_FINALIZE", "finalize")
    assert ops.finalize_stage(128, 4, 12, 148) == "finalize"
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_FINALIZE", "finalize_mlp")
    assert (
        ops.finalize_stage(128, 16, 48, 160) == "finalize_mlp"
    )  # a TP1 grid pinned to the one-wave kernel (experiments)
    with pytest.raises(ValueError, match="group"):
        ops.finalize_stage(3, 2, 4, 148)  # pinned outside the group contract: never launched
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_FINALIZE", "mlp")
    with pytest.raises(ValueError, match="MILES_GDN_CHUNK_TRAIN_FINALIZE"):
        ops.finalize_stage(128, 4, 12, 148)


def test_finalize_stages_are_exported_with_one_plan_and_the_backward_launches_the_selected_one(monkeypatch):
    """Both finalize records exist for this architecture with the same argument plan and four-warp block; the backward of
    the TP4 micro-batch launches ``finalize_mlp`` (492 CTAs, group 3) and a group-2 layout launches ``finalize``."""
    _require_blackwell()
    arch = ops.device_arch(DEVICE)
    base, mlp = ops.MODULES["finalize"][arch], ops.MODULES["finalize_mlp"][arch]
    assert base["schedule"] == "gdn_chunk_bwd_finalize" and mlp["schedule"] == "gdn_chunk_bwd_finalize_mlp"
    assert base["block"] == mlp["block"] == [128, 1, 1] and base["arg_plan"] == mlp["arg_plan"]
    assert base["kernel_symbol"] != mlp["kernel_symbol"]
    launched = []
    real_launcher = ops.launcher

    def spy(stage, arch_, names):
        launched.append(stage)
        return real_launcher(stage, arch_, names)

    monkeypatch.setattr(ops, "launcher", spy)
    sm_count = ops._device_props(DEVICE.index)[0]
    for layout, heads, v_heads in ((LAYOUT, 4, 12), (LAYOUT, 2, 4)):
        launched.clear()
        inputs = _inputs(layout, num_k_heads=heads, num_v_heads=v_heads, seed=5)
        cu = torch.tensor(_cu(layout), device="cuda", dtype=torch.int32)
        _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(layout))
        finalize_launches = [s for s in launched if s in ops.FINALIZE_STAGES]
        expected = ops.finalize_stage(_chunks(layout), heads, v_heads, sm_count)
        assert finalize_launches == [expected], (layout, heads, v_heads, finalize_launches)
    assert expected == "finalize"  # the group-2 layout
    assert ops.finalize_stage(_chunks(LAYOUT), 4, 12, sm_count) == "finalize_mlp"


def test_warmup_threshold_env_override(monkeypatch):
    monkeypatch.delenv("MILES_GDN_CHUNK_TRAIN_WARMUP_LOG2", raising=False)
    assert ops.warmup_thresh_log2() == ops.WARMUP_THRESH_LOG2
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_WARMUP_LOG2", "-inf")
    assert ops.warmup_thresh_log2() == float("-inf")


@pytest.mark.parametrize("sm_count", (B200_SMS, GB300_SMS))
def test_segment_planner_reproduces_the_modelled_optima(sm_count):
    # one 8192-token sequence (129 padded chunks): three 43-chunk segments on both scans (fwd_h at two CTAs per SM:
    # 54.2 chunk steps vs 55.6 for six -- the forced-L sweep measured L = 22 / 43 as a tie on fwd_h and L = 43 as the
    # best f+b arm; dhu at one CTA per SM: six segments would run two waves); 16384 tokens: 43 / 86
    assert ops.plan_segments(129, 1, HV12_CTAS, sm_count, 2) == (43, pytest.approx(54.178))
    assert ops.plan_segments(129, 1, HV12_CTAS, sm_count, 1) == (43, pytest.approx(54.178))
    assert ops.plan_segments(257, 1, HV12_CTAS, sm_count, 2) == (43, pytest.approx(89.1748))
    assert ops.plan_segments(257, 1, HV12_CTAS, sm_count, 1) == (86, pytest.approx(97.178))
    # the real micro-batch 8481 + 223 (pad) tokens on the bound (137 + 1 chunks): the pad's CTAs make every grid above
    # one CTA per SM co-resident, so fwd_h takes six segments (23) and dhu at one CTA per SM three (46)
    assert ops.plan_segments(138, 2, HV12_CTAS, sm_count, 2)[0] == 23
    assert ops.plan_segments(138, 2, HV12_CTAS, sm_count, 1)[0] == 46
    # balanced 4 x 4096 / 8 x 1024 / 4 x 1024 layouts share the bound of one long sequence (planner in isolation; the
    # host route prices the exact layout and production guards 8 x 1024 and the 4 x 4096 dhu scan to L = 0)
    assert ops.plan_segments(260, 4, HV12_CTAS, sm_count, 2)[0] == 43
    assert ops.plan_segments(136, 8, HV12_CTAS, sm_count, 2)[0] == 26
    assert ops.plan_segments(68, 4, HV12_CTAS, sm_count, 2)[0] == 11
    # TP1 (192 CTAs per segment): three segments run two rounds of the resident set (576 CTAs at 3.2x) and still beat
    # the serial scan's 192 co-resident CTAs by 24 % where two CTAs fit one SM; at one CTA per SM the same three
    # segments run four sequential waves of 43 steps (172 + 11.5 + 7.35) against the serial scan's two waves of 129
    # (R6: admissible up to MAX_SCAN_WAVES_SERIAL; the joint planner still prefers the (64,1) split there, 126.3)
    assert ops.plan_segments(129, 1, HV48_CTAS, sm_count, 2) == (43, pytest.approx(157.1996))
    assert ops.plan_segments(129, 1, HV48_CTAS, sm_count, 1) == (43, pytest.approx(190.834))
    # documented limitation: a balanced 4 x 1024 layout at four value heads is priced as one long sequence on the bound
    assert ops.plan_segments(68, 4, 4 * (ops.HEAD_DIM // 32), sm_count, 2)[0] == 13


def _split_cost(chunks, seg_len, per_segment, sm_count, residency):
    # the planner's price of segment length ``seg_len`` on ``chunks``: pass-2 makespan over every sequence's segments,
    # warm-up pre-scan of the consumed segments' value blocks, stitch launch floor + one copy-only step per boundary
    splits = tuple(max(1, -(-n // seg_len)) for n in chunks)
    return (
        ops._scan_makespan(chunks, splits, per_segment, sm_count, residency)
        + ops.PRESCAN_WARMUP_CHUNKS
        * ops.PRESCAN_STEP_FACTOR
        * ops._wave_factor(sum(k - 1 for k in splits) * per_segment, sm_count, residency)
        + ops.STITCH_FIXED_CHUNKS
        + ops.STITCH_COPY_CHUNKS * (max(splits) - 1)
    )


def _admissible_segment_lens(chunks, per_segment, sm_count, residency):
    longest = max(chunks)
    # the calibration bound restated: two rounds of the two-CTA resident set when co-resident, four waves at one CTA per SM
    limit = (ops.MAX_SCAN_ROUNDS * 2 if residency >= 2 else ops.MAX_SCAN_WAVES_SERIAL) * sm_count
    assert limit == ops.split_cta_limit(sm_count, residency)
    return {
        -(-longest // s)
        for s in range(2, longest // ops.MIN_SEGMENT_CHUNKS + 1)
        if ops.MIN_SEGMENT_CHUNKS <= -(-longest // s) < longest
        and sum(max(1, -(-n // -(-longest // s))) for n in chunks) * per_segment <= limit
    }


@pytest.mark.parametrize("residency", (2, 1))
def test_segment_planner_only_splits_for_a_modelled_gain(residency):
    per_segment, sm_count = HV12_CTAS, B200_SMS
    for bound in range(2, 400, 7):
        for num_seqs in (1, 2, 5):
            if bound < num_seqs:
                continue
            chunks = ops.bound_layout_chunks(bound, num_seqs)
            seg_len, cost = ops.plan_segments(bound, num_seqs, per_segment, sm_count, residency)
            serial = ops._scan_makespan(chunks, (1,) * num_seqs, per_segment, sm_count, residency)
            # independent argmin over every admissible segment length (MIN_SEGMENT_CHUNKS and MAX_SCAN_ROUNDS filters)
            admissible = {
                length: _split_cost(chunks, length, per_segment, sm_count, residency)
                for length in _admissible_segment_lens(chunks, per_segment, sm_count, residency)
            }
            if seg_len == 0:
                assert cost == serial
                # the converse of the test name: no admissible split beats the serial scan by the required 5 %
                if admissible:
                    assert min(admissible.values()) >= ops.SPLIT_GAIN_THRESHOLD * serial
            else:
                assert ops.MIN_SEGMENT_CHUNKS <= seg_len < max(chunks)
                assert cost < ops.SPLIT_GAIN_THRESHOLD * serial
                assert cost == admissible[seg_len] == min(admissible.values())
                # the strict < in the search keeps the first minimum: the largest L (smallest S) among equal costs
                assert seg_len == max(length for length, other in admissible.items() if other == cost)


@pytest.mark.parametrize("sm_count", (B200_SMS, GB300_SMS))
def test_segment_planner_prices_the_exact_host_layout(sm_count):
    chunks = lambda *lens: tuple(-(-t // ops.CHUNK) for t in lens)  # noqa: E731 - per-sequence chunk counts (ceil)
    assert ops.bound_layout_chunks(138, 2) == (137, 1)
    for bound, n in ((129, 1), (257, 1), (138, 2), (260, 4), (136, 8)):
        assert ops.plan_segments(bound, n, HV12_CTAS, sm_count, 2) == ops.plan_segments(
            bound, n, HV12_CTAS, sm_count, 2, seq_chunks=ops.bound_layout_chunks(bound, n)
        )
    # one dominant sequence plus the 512-alignment pseudo-sequence: the pad's 48 CTAs push every fwd_h grid above one
    # CTA per SM (co-resident throughout), so five segments beat three (61.3 vs 83.2 chunk steps for 8481 + 223);
    # dhu at one CTA per SM keeps three (60.2 vs 73.6 for five)
    assert ops.plan_segments(138, 2, HV12_CTAS, sm_count, 2, seq_chunks=chunks(8192, 512))[0] == 26
    assert ops.plan_segments(138, 2, HV12_CTAS, sm_count, 2, seq_chunks=chunks(8481, 223)) == (
        27,
        pytest.approx(61.2748),
    )
    assert ops.plan_segments(138, 2, HV12_CTAS, sm_count, 1, seq_chunks=chunks(8481, 223)) == (
        45,
        pytest.approx(60.178),
    )
    assert ops.segment_bounds(133, 27) == [0, 26, 53, 79, 106, 133]
    assert ops.segment_bounds(133, 45) == ops.segment_bounds(133, 46) == [0, 44, 88, 133]
    assert ops.plan_segments(260, 2, HV12_CTAS, sm_count, 2, seq_chunks=chunks(16384, 512))[0] == 52
    assert ops.plan_segments(260, 2, HV12_CTAS, sm_count, 1, seq_chunks=chunks(16384, 512))[0] == 86
    # two long sequences plus the pseudo-sequence (69 + 55 + 6 chunks): the exact layout splits fwd_h at L = 23
    # (3 + 3 + 1 segments, 59.9 vs 69 chunk steps) and keeps dhu serial (one CTA per SM); the bound splits 22 / 43
    assert ops.plan_segments(131, 3, HV12_CTAS, sm_count, 2)[0] == 22
    assert ops.plan_segments(131, 3, HV12_CTAS, sm_count, 1)[0] == 43
    assert ops.plan_segments(131, 3, HV12_CTAS, sm_count, 2, seq_chunks=chunks(4360, 3491, 341)) == (
        23,
        pytest.approx(59.8748),
    )
    assert ops.plan_segments(131, 3, HV12_CTAS, sm_count, 1, seq_chunks=chunks(4360, 3491, 341)) == (0, 69.0)
    # packs: 44 / 44 / 39 / 3 chunks (192 CTAs, co-resident even unsplit) split fwd_h at 22; the real five-sequence
    # micro-batch (38/38/38/15/3 chunks, 240 CTAs) stays serial on both scans (the R3 pre-fit planner split it at
    # L = 19 and measured +5 % on the forward); 4 x 4096 splits fwd_h at L = 22 (12 segments, two rounds at 3.2x: 90
    # vs 102.4 chunk steps for 192 co-resident CTAs); its one-CTA-per-SM dhu scan in (32,2) terms also splits at 22
    # since R6 (576 CTAs, four waves: 88 + 11.5 + 7.35 = 106.8 vs 128 serial; the joint planner takes the (64,1)
    # split, 73.0); 4 x 1024 at HV 4 stays serial
    assert ops.plan_segments(132, 4, HV12_CTAS, sm_count, 2, seq_chunks=chunks(2802, 2771, 2476, 143)) == (
        22,
        pytest.approx(48.878),
    )
    assert ops.plan_segments(133, 5, HV12_CTAS, sm_count, 2, seq_chunks=chunks(2384, 2382, 2369, 902, 155)) == (
        0,
        pytest.approx(60.8),
    )
    assert ops.plan_segments(260, 4, HV12_CTAS, sm_count, 2, seq_chunks=chunks(4096, 4096, 4096, 4096)) == (
        22,
        pytest.approx(89.9996),
    )
    assert ops.plan_segments(260, 4, HV12_CTAS, sm_count, 1, seq_chunks=chunks(4096, 4096, 4096, 4096)) == (
        22,
        pytest.approx(106.834),
    )
    assert ops.plan_segments(130, 2, 24 * (ops.HEAD_DIM // 32), sm_count, 2, seq_chunks=chunks(4096, 4096)) == (
        22,
        pytest.approx(89.9996),
    )
    assert ops.plan_segments(68, 4, 16, sm_count, 2, seq_chunks=chunks(1024, 1024, 1024, 1024)) == (0, 16.0)
    # the wiring LAYOUT at HV 4 splits both scans at 18 on the exact layout
    assert ops.plan_segments(125, 4, 16, sm_count, 2, seq_chunks=chunks(3339, 2802, 1340, 263)) == (
        18,
        pytest.approx(29.178),
    )
    # R6: one-CTA-per-SM (64,1) scans priced in their own terms (per_segment = HV x 2, step factor 1.27).  Two
    # 4096-token sequences at HV 48 and four at HV 24 run 192 serial CTAs x 64 steps = two waves (the second 26-30 %
    # full, 162.6); 22-chunk segments run 576 CTAs in four waves of 22 (111.8 + 14.6 + 7.35 = 133.7).  Four 4096-token
    # sequences at HV 48 (384 CTAs, three waves) and eight 1024-token sequences at HV 12 have no split that saves a
    # wave and stay serial; the device route (bound layouts 129 + 1 / 257 + 3 at HV 48) splits at 43 / 86.
    assert ops.plan_segments(130, 2, 96, sm_count, 1, seq_chunks=chunks(4096, 4096), step_factor=1.27) == (
        22,
        pytest.approx(133.69468),
    )
    assert ops.plan_segments(260, 4, 48, sm_count, 1, seq_chunks=chunks(4096, 4096, 4096, 4096), step_factor=1.27) == (
        22,
        pytest.approx(133.69468),
    )
    assert ops.plan_segments(260, 4, 96, sm_count, 1, seq_chunks=chunks(4096, 4096, 4096, 4096), step_factor=1.27) == (
        0,
        pytest.approx(243.84),
    )
    assert ops.plan_segments(136, 8, 24, sm_count, 1, seq_chunks=chunks(*([1024] * 8)), step_factor=1.27) == (
        0,
        pytest.approx(40.64),
    )
    assert ops.plan_segments(130, 2, 96, sm_count, 1, step_factor=1.27) == (43, pytest.approx(127.56312))
    assert ops.plan_segments(260, 4, 96, sm_count, 1, step_factor=1.27) == (86, pytest.approx(346.00312))


def test_host_route_prices_the_exact_layout():
    _require_blackwell()
    ops._host_chunk_meta.cache_clear()
    arch = ops.device_arch(DEVICE)
    index = DEVICE.index
    # two long sequences + the 512-alignment pseudo-sequence: the device route (chunk bound only) prices one 129-chunk
    # chain (22 / 43), the host route the exact 69 + 55 + 6 layout (fwd_h 23: 3 + 3 + 1 segments; dhu serial)
    # (R3, C6: the joint plan moves both scans onto the (64,1) block, see the generator's
    # test_scan_plan_prices_configuration_switches_and_splits; the exact layout and the bound differ in L)
    P = ops.ScanPlan
    pack = (4360, 3491, 341)
    assert ops.scan_plans(sum(pack), 3, 12, index, arch) == (P((64, 1), 33), P((64, 1), 22))
    assert ops.scan_plans(sum(pack), 3, 12, index, arch, pack) == (P((64, 1), 12), P((64, 1), 23))
    assert ops.segment_lens(sum(pack), 3, 12, index, arch, pack) == (12, 23)
    meta = ops.chunk_meta_from_host(_cu(pack), DEVICE, 12, arch=arch)
    assert meta.fwd_segments.scan_config == meta.bwd_segments.scan_config == (64, 1)
    assert meta.fwd_segments.segment_len == 12 and meta.bwd_segments.segment_len == 23
    assert meta.fwd_segments.num_segments == 12 and meta.bwd_segments.num_segments == 7
    assert meta.fwd_segments.seg_chunk_start.tolist() == [0, 11, 23, 34, 46, 57, 69, 80, 91, 102, 113, 124, 130]
    assert meta.bwd_segments.seg_chunk_start.tolist() == [0, 23, 46, 69, 87, 105, 124, 130]
    # one dominant sequence + pad: the device route (bound 137 + 1) picks 28 / 23, the exact layout 27 / 27 (five
    # segments per scan on the (64,1) block: 168 CTAs incl. the pad's, inside one CTA per SM)
    mb00 = (8481, 223)
    assert ops.scan_plans(sum(mb00), 2, 12, index, arch) == (P((64, 1), 28), P((64, 1), 23))
    assert ops.scan_plans(sum(mb00), 2, 12, index, arch, mb00) == (P((64, 1), 27), P((64, 1), 27))
    host = ops.chunk_meta_from_host(_cu(mb00), DEVICE, 12, arch=arch)
    assert host.fwd_segments.segment_len == 27 and host.fwd_segments.num_segments == 6  # 5 segments + the pad sequence
    assert host.fwd_segments.seg_chunk_start.tolist() == [0, 26, 53, 79, 106, 133, 137]
    assert host.bwd_segments.segment_len == 27 and host.bwd_segments.seg_chunk_start.tolist() == [
        0,
        26,
        53,
        79,
        106,
        133,
        137,
    ]
    # layouts whose (32,2) grid is small keep the preferred block: the wiring layout at HV 4 splits both scans at 18
    assert ops.scan_plans(sum(LAYOUT), len(LAYOUT), 4, index, arch, LAYOUT) == (P((32, 2), 18), P((32, 2), 18))
    # a pinned configuration is planned for: (64,1) at 1x8192 / HV 12 -> six 22-chunk segments on both scans
    ops.FORCED_SCAN_CONFIG = (64, 1)
    try:
        assert ops.scan_plans(8192, 1, 12, index, arch) == (P((64, 1), 22), P((64, 1), 22))
    finally:
        ops.FORCED_SCAN_CONFIG = None
    assert ops.scan_plans(8192, 1, 12, index, arch) == (
        P((64, 1), 22),
        P((64, 1), 22),
    )  # the joint plan picks it unpinned too
    assert ops.scan_plans(8192, 1, 48, index, arch) == (
        P((64, 1), 43),
        P((64, 1), 43),
    )  # TP1: 288 CTAs in three segments


def test_scan_plans_are_memoised_per_layout_and_pins(monkeypatch):
    # mirror of the generator's test: the memo key carries the layout, the device and the pins (a pin change re-plans)
    arch = ops.device_arch(DEVICE)
    P = ops.ScanPlan
    first = ops.scan_plans(8192, 1, 12, DEVICE.index, arch)
    assert first == (P((64, 1), 22), P((64, 1), 22))
    assert ops.scan_plans(8192, 1, 12, DEVICE.index, arch) is first
    assert ops._scan_plans_memo.cache_info().hits == 1 and ops._scan_plans_memo.cache_info().misses == 1
    monkeypatch.setattr(ops, "FORCED_SCAN_CONFIG", (32, 2))
    assert ops.scan_plans(8192, 1, 12, DEVICE.index, arch) == (P((32, 2), 43), P((32, 2), 43))
    monkeypatch.setattr(ops, "FORCED_SCAN_CONFIG", None)
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS", "0")
    assert ops.scan_plans(8192, 1, 12, DEVICE.index, arch) == (P((32, 2), 0), P((32, 2), 0))
    monkeypatch.delenv("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS")
    assert ops.scan_plans(8192, 1, 12, DEVICE.index, arch) is first
    assert ops._scan_plans_memo.cache_info().misses == 3


def test_segment_len_env_override_wins_over_the_planner(monkeypatch):
    # the pin is validated at plan time against the stages rendered for the device's arch (R3: a pin on a layout or
    # arch without a pre-scan stage is dropped with a warning, see test_segment_len_override_is_validated_at_plan_time)
    _require_blackwell()
    arch = ops.device_arch(DEVICE)
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", None)
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS", "7")
    assert ops.segment_lens(8192, 1, 12, DEVICE.index, arch) == (7, 7)
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS", "0")
    assert ops.segment_lens(8192, 1, 12, DEVICE.index, arch) == (0, 0)


def test_segment_len_override_is_validated_at_plan_time(monkeypatch):
    _require_blackwell()
    arch = ops.device_arch(DEVICE)
    # a five-sequence TP4 layout runs the (64,1) configuration on the dhu scan (no pre-scan stage) and (32,2) on fwd_h:
    # the pin is honoured where a pre-scan exists and dropped with a warning naming the layout where none does
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS", "43")
    assert ops.choose_register_scan_config("dhu", 5, 12, DEVICE.index, arch) == (64, 1)
    # (64,1) has pre-scan stages since R3 (C6): the pin is honoured on both scans
    assert ops.segment_lens(16384, 5, 12, DEVICE.index, arch) == (43, 43)
    # (32,1) has none: pinned through the environment, the segment pin is dropped with a warning naming the layout
    monkeypatch.setenv("MILES_GDN_CHUNK_TRAIN_SCAN_CFG", "32,1")
    with pytest.warns(RuntimeWarning, match=r"forced segment length 43 ignored for the dhu scan .*\(32, 1\)"):
        assert ops.segment_lens(16384, 5, 12, DEVICE.index, arch) == (0, 0)


# --------------------------------------------------------------------------- #
# per-architecture stage selection (dqkwg, wy_bwd, wy_bwd_bf16; the WY pullback is the tcgen05 kernel on both Blackwell parts)

PER_ARCH_STAGES = {
    # stage: {arch: (schedule the registry names, block)}; the eight-warp dqkwg is selected on SM103a only, the
    # tcgen05 + swapAB WY pullback on SM100a and SM103a
    "dqkwg": {"sm_100a": ("gdn_chunk_bwd_dqkwg", 128), "sm_103a": ("gdn_chunk_bwd_dqkwg_w8", 256)},
    "wy_bwd": {"sm_100a": ("gdn_chunk_bwd_wy_tc", 256), "sm_103a": ("gdn_chunk_bwd_wy_tc", 256)},
    "wy_bwd_bf16": {"sm_100a": ("gdn_chunk_bwd_wy_bf16_tc", 256), "sm_103a": ("gdn_chunk_bwd_wy_bf16_tc", 256)},
}


def test_per_arch_stages_launch_the_kernel_selected_for_this_architecture():
    """``dqkwg`` / ``wy_bwd`` / ``wy_bwd_bf16`` are the stages whose kernel is selected per architecture: the registry
    the host loads names the schedule it was rendered from and carries that kernel's block / dynamic shared
    memory, and ``kernel()`` launches exactly that record on the current device (eight-warp dqkwg on SM103a only,
    tcgen05 WY pullback on both Blackwell parts)."""
    _require_blackwell()
    arch = ops.device_arch(DEVICE)
    for stage, per_arch in PER_ARCH_STAGES.items():
        schedule, threads = per_arch[arch]
        record = ops.MODULES[stage][arch]
        assert record["schedule"] == schedule, (stage, arch, record["schedule"])
        assert record["block"] == [threads, 1, 1], (stage, arch, record["block"])
        assert record["dynamic_smem_bytes"] > 0, stage
        native = ops.kernel(stage, arch)
        assert native.arch == arch and native.record is record, stage
    # every other stage is rendered from one schedule for both architectures: its record never names a variant
    for stage, per_arch in ops.MODULES.items():
        if stage not in PER_ARCH_STAGES and "schedule" in per_arch[arch]:
            assert not per_arch[arch]["schedule"].endswith("_w8"), stage


# --------------------------------------------------------------------------- #
# host-side caches (round R6-7): workspace pool, chunk-table front cache, positional launchers
# --------------------------------------------------------------------------- #
FORWARD_ROLES = {"f_zero_state", "f_seg_state", "f_seg_flag"}
BACKWARD_ROLES = {
    "b_dv_local",
    "b_dh",
    "b_dv2",
    "b_zero_state",
    "b_seg_state",
    "b_seg_flag",
    "b_dh_end",
    "b_dq_hv",
    "b_dk_hv",
    "b_dw",
    "b_dg1",
    "b_dk2",
    "b_dg2",
}


def _pooled_tensors(ws):
    return [t for table in ws.tables.values() for buffers in table.values() for t in buffers.values()]


def _pooled_roles(ws):
    return {role for table in ws.tables.values() for role in table}


def _storage_ranges(tensors):
    return [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors if t is not None and t.numel() > 0
    ]


def _overlaps(a, b):
    return any(x0 < y1 and y0 < x1 for x0, x1 in a for y0, y1 in b)


def _expected_pool_misses(meta):
    # fwd: 0-element final_state placeholder (+ seg_state, seg_flag when the forward scan splits; the stitch's start-state
    # scratch is gone, the split scan's prologue reads the maps); bwd: dv_local, dh, dv2, 0-element dh0 placeholder, dq_hv,
    # dk_hv, dw, dg1, dk2, dg2 (+ seg_state, seg_flag, dh_end when the backward scan splits)
    split = [s.segment_len > 0 and s.num_segments > meta.num_seqs for s in (meta.fwd_segments, meta.bwd_segments)]
    return (1 + 2 * split[0]) + (10 + 3 * split[1])


def test_workspace_pool_keys_separate_role_shape_dtype_and_stream():
    ws = ops.Workspaces()
    index = DEVICE.index
    default = ws.table(index)
    assert ws.table(index) is default, "one table per (device, current stream)"
    with torch.cuda.stream(torch.cuda.Stream(device=DEVICE)):
        other = ws.table(index)
    assert other is not default, "another stream gets its own table (no cross-stream reuse)"
    a = ws.take(default, "b_x", (4, 8), torch.float32, DEVICE)
    assert ws.take(default, "b_x", (4, 8), torch.float32, DEVICE) is a and (ws.hits, ws.misses) == (1, 1)
    assert (
        ws.take(default, "b_x", torch.Size([4, 8]), torch.float32, DEVICE) is a
    ), "torch.Size and tuple shapes share a key"
    assert ws.take(default, "b_x", (4, 8), torch.bfloat16, DEVICE) is not a, "dtype is part of the key"
    assert ws.take(default, "b_x", (8, 4), torch.float32, DEVICE) is not a, "exact shape is part of the key"
    assert ws.take(default, "b_y", (4, 8), torch.float32, DEVICE) is not a, "role is part of the key"
    assert ws.take(other, "b_x", (4, 8), torch.float32, DEVICE) is not a, "stream is part of the key"
    # the oldest (shape, dtype) of a role is evicted beyond WORKSPACE_SHAPES_PER_ROLE entries
    for n in range(ops.WORKSPACE_SHAPES_PER_ROLE):
        ws.take(default, "b_x", (16 + n, 1), torch.float32, DEVICE)
    assert len(default["b_x"]) == ops.WORKSPACE_SHAPES_PER_ROLE
    misses = ws.misses
    assert ws.take(default, "b_x", (4, 8), torch.float32, DEVICE) is not a and ws.misses == misses + 1
    # the GDN_HOST_CACHE=0 arm never reuses
    fresh = ops.FreshWorkspaces()
    t = fresh.table(index)
    assert fresh.take(t, "b_x", (4, 8), torch.float32, DEVICE) is not fresh.take(
        t, "b_x", (4, 8), torch.float32, DEVICE
    )
    assert fresh.hits == 0 and fresh.misses == 2
    ws.clear()
    assert not ws.tables


def test_workspace_pool_hits_after_the_first_call_and_results_match_the_uncached_arm(monkeypatch):
    _require_blackwell()
    inputs = _inputs(LAYOUT, seed=5)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    kwargs = dict(cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    pooled = ops.Workspaces()
    monkeypatch.setattr(ops, "WORKSPACES", pooled)
    first = _step(inputs, **kwargs)
    torch.cuda.synchronize()
    meta = ops.chunk_meta_from_host(_cu(LAYOUT), DEVICE, 4)
    expected = _expected_pool_misses(meta)
    assert (pooled.hits, pooled.misses) == (0, expected), (pooled.hits, pooled.misses, expected)
    assert _pooled_roles(pooled) == (
        FORWARD_ROLES if meta.fwd_segments.num_segments > meta.num_seqs else {"f_zero_state"}
    ) | (
        BACKWARD_ROLES
        if meta.bwd_segments.num_segments > meta.num_seqs
        else BACKWARD_ROLES - {"b_seg_state", "b_seg_flag", "b_dh_end"}
    )
    for layer in range(3):  # the next layers of the micro-batch reuse every buffer: no further allocation
        again = _step(inputs, **kwargs)
        for name in first:
            assert torch.equal(first[name], again[name]), f"{name}: layer {layer + 1} differs with pooled workspaces"
    torch.cuda.synchronize()
    assert (pooled.hits, pooled.misses) == (3 * expected, expected), (pooled.hits, pooled.misses)
    # the GDN_HOST_CACHE=0 arm (fresh torch.empty per scratch buffer) is bit-identical to the pooled arm
    monkeypatch.setattr(ops, "WORKSPACES", ops.FreshWorkspaces())
    fresh = _step(inputs, **kwargs)
    for name in first:
        assert torch.equal(first[name], fresh[name]), f"{name}: pooled vs fresh workspaces differ"
    # and so is the tape-free recompute backward with pooled workspaces (forward roles reused inside the backward call)
    monkeypatch.setattr(ops, "WORKSPACES", ops.Workspaces())
    monkeypatch.setattr(ops, "SAVE_STATE", False)
    recomputed = _step(inputs, **kwargs)
    for name in first:
        assert torch.equal(first[name], recomputed[name]), f"{name}: recompute backward with pooled workspaces differs"


def test_pooled_workspaces_never_alias_saved_returned_or_gradient_tensors(monkeypatch):
    _require_blackwell()
    inputs = _inputs(LAYOUT, seed=6)
    q, k, v, g, beta, do = inputs
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    ws = ops.Workspaces()
    monkeypatch.setattr(ops, "WORKSPACES", ws)
    leaves = _leaves(inputs)
    out, final_state = ops.chunk_gated_delta_rule(
        *leaves, use_qk_l2norm_in_kernel=True, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT)
    )
    assert out.shape == v.shape and out.dtype == v.dtype and final_state is None
    saved = list(out.grad_fn.saved_tensors)
    assert len(saved) >= 10 + 4 * int(ops.SAVE_STATE)
    pooled = _storage_ranges(_pooled_tensors(ws))
    assert pooled, "the forward must have pooled at least its placeholder"
    assert not _overlaps(
        pooled, _storage_ranges(saved + [out])
    ), "a pooled workspace aliases a saved or returned tensor"
    assert _pooled_roles(ws) <= FORWARD_ROLES, _pooled_roles(ws)
    out.backward(do)
    grads = [t.grad for t in leaves]
    assert [tuple(t.shape) for t in grads] == [tuple(t.shape) for t in (q, k, v, g, beta)]
    assert [t.dtype for t in grads] == [q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype]
    pooled = _storage_ranges(_pooled_tensors(ws))
    assert not _overlaps(pooled, _storage_ranges(saved + [out] + grads)), "a pooled workspace aliases a gradient"
    assert _pooled_roles(ws) <= FORWARD_ROLES | BACKWARD_ROLES, _pooled_roles(ws)
    # every pooled buffer lives on the device and the stream the call ran on
    assert set(ws.tables) == {(DEVICE.index, torch.cuda.current_stream(DEVICE.index).cuda_stream)}
    assert all(t.device == DEVICE for t in _pooled_tensors(ws))


def test_chunk_meta_front_cache_hits_per_layer_and_keys_on_layout_heads_device_and_pins(monkeypatch):
    _require_blackwell()
    arch = ops.device_arch(DEVICE)
    ops._host_chunk_meta.cache_clear()
    before, cu = ops._HOST_META_BUILDS, torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    metas = [ops._chunk_meta(cu, _cu(LAYOUT), 1, sum(LAYOUT), DEVICE, arch, 4) for _ in range(48)]
    assert ops._HOST_META_BUILDS == before + 1 and all(m is metas[0] for m in metas)
    assert ops._META_FRONT_HITS == 47, ops._META_FRONT_HITS
    # the head count is part of the key: a miss (whether the planner then picks the same tables is the planner's business)
    hits = ops._META_FRONT_HITS
    m12 = ops._chunk_meta(cu, _cu(LAYOUT), 1, sum(LAYOUT), DEVICE, arch, 12)
    assert (
        ops._META_FRONT_HITS == hits
        and m12.fwd_segments.scan_config
        == ops.scan_plans(sum(LAYOUT), len(LAYOUT), 12, DEVICE.index, arch, LAYOUT)[0].config
    )
    assert (
        ops._chunk_meta(cu, _cu(LAYOUT), 1, sum(LAYOUT), DEVICE, arch, 12) is m12 and ops._META_FRONT_HITS == hits + 1
    )
    builds = ops._HOST_META_BUILDS
    other = ops._chunk_meta(
        torch.tensor(_cu((8192, 512)), device="cuda", dtype=torch.int32), _cu((8192, 512)), 1, 8704, DEVICE, arch, 4
    )
    assert other is not metas[0] and ops._HOST_META_BUILDS == builds + 1
    # a CPU tensor copy of the boundaries bypasses the front cache but resolves to the same memoised table
    hits, builds = ops._META_FRONT_HITS, ops._HOST_META_BUILDS
    assert (
        ops._chunk_meta(cu, torch.tensor(_cu(LAYOUT), dtype=torch.int64), 1, sum(LAYOUT), DEVICE, arch, 4) is metas[0]
    )
    assert ops._HOST_META_BUILDS == builds and ops._META_FRONT_HITS == hits
    # a plan pin re-plans: the pins are part of the key (det_r2.py sets FORCED_SEGMENT_LEN between arms)
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", 4)
    pinned = ops._chunk_meta(cu, _cu(LAYOUT), 1, sum(LAYOUT), DEVICE, arch, 4)
    assert pinned is not metas[0] and pinned.fwd_segments.segment_len == 4 and ops._HOST_META_BUILDS == builds + 1
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", None)
    assert ops._chunk_meta(cu, _cu(LAYOUT), 1, sum(LAYOUT), DEVICE, arch, 4) is metas[0]
    # equal-length batches (cu_seqlens None) are cached under their own layout key
    equal = ops._chunk_meta(None, None, 2, 1024, DEVICE, arch, 4)
    assert ops._chunk_meta(None, None, 2, 1024, DEVICE, arch, 4) is equal and equal.num_seqs == 2
    # the device route (no host copy) never enters the front cache: one meta launch per call, as before
    hits = ops._META_FRONT_HITS
    dev = ops._chunk_meta(cu, None, 1, sum(LAYOUT), DEVICE, arch, 4)
    assert dev.seq_lens is None and ops._META_FRONT_HITS == hits
    ops.host_cache_clear()
    assert ops._META_FRONT_HITS == 0 and not ops._META_FRONT


def test_positional_launchers_match_every_exported_plan():
    _require_blackwell()
    from miles.kernels.attention.gdn.cake import _jit

    arch = ops.device_arch(DEVICE)
    stages = sorted(stage for stage, per_arch in ops.MODULES.items() if arch in per_arch and stage != "meta")
    assert set(stages) <= set(ops.LAUNCH_ARGS), sorted(set(stages) - set(ops.LAUNCH_ARGS))
    for stage in stages:
        names = ops.LAUNCH_ARGS[stage]
        run = _jit.launcher(stage, arch, names)
        assert _jit.launcher(stage, arch, names) is run, "one bound launcher per (stage, arch, order)"
        plan = tuple(name for kind, name in ops.MODULES[stage][arch]["arg_plan"] if kind != "grid")
        assert sorted(run.plan) == sorted(names) == sorted(plan), stage
        if run.plan != names:  # still correct (the permuting path maps names to plan slots); the fast path is lost
            warnings.warn(
                f"{stage}: the call-site order differs from the exported plan order {plan!r}",
                RuntimeWarning,
                stacklevel=1,
            )
        assert tuple(kind for kind, _ in ops.MODULES[stage][arch]["arg_plan"][-3:]) == ("grid",) * 3
        assert ops.kernel(stage, arch).record is ops.MODULES[stage][arch]
    native = ops.kernel("stitch", arch)
    with pytest.raises(KeyError):
        native.launcher(("seg_state",))
    with pytest.raises(KeyError):
        native.launcher(ops.LAUNCH_ARGS["stitch"] + ("extra",))
    with pytest.raises(TypeError):
        _jit.launcher("stitch", arch, ops.LAUNCH_ARGS["stitch"])(1, 1, 1)
    # a permuted order binds through the permuting path and maps names to the same plan slots
    permuted = tuple(reversed(ops.LAUNCH_ARGS["stitch"]))
    run = native.launcher(permuted)
    assert run.names == permuted and run.plan == ops.LAUNCH_ARGS["stitch"]


def test_forward_passes_batched_shapes_through_and_backward_returns_input_shapes():
    _require_blackwell()
    q, k, v, g, beta, do = _inputs(LAYOUT, seed=7)
    meta = ops.chunk_meta_from_host(_cu(LAYOUT), DEVICE, 4)
    flat = ops.gdn_chunk_forward(q[0], k[0], v[0], g[0], beta[0], meta=meta, normalize_qk=True)
    batched = ops.gdn_chunk_forward(q, k, v, g, beta, meta=meta, normalize_qk=True)
    assert flat["output"].shape == v[0].shape and batched["output"].shape == v.shape
    assert torch.equal(
        flat["output"], batched["output"][0]
    ), "the [T, H, D] and [1, T, H, D] forms run the same kernels on the same bytes"
    assert (
        batched["total_tokens"] == sum(LAYOUT)
        and batched["qn"].shape == q.shape
        and batched["w"].shape == (sum(LAYOUT), 4, 128)
    )
    gf = ops.gdn_chunk_backward(flat, do[0], dbeta_dtype=beta.dtype)
    gb = ops.gdn_chunk_backward(dict(batched, g_shape=g.shape, beta_shape=beta.shape), do, dbeta_dtype=beta.dtype)
    assert (
        gb["dq"].shape == q.shape
        and gb["dv"].shape == v.shape
        and gb["dg"].shape == g.shape
        and gb["dbeta"].shape == beta.shape
    )
    for name in ("dq", "dk", "dv", "dg", "dbeta"):
        assert torch.equal(gf[name], gb[name][0]), name
