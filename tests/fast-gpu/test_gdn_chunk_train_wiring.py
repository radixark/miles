"""Host wiring of the deterministic chunked GDN kernels (``miles_plugins/models/gdn_chunk_train``).

One exact chunk table per micro-batch layout (shared by every layer, no device-to-host synchronisation), launch
counts, and the bit-identical tape / recompute backward.  Needs a CUDA device; the kernel tests need Blackwell
(SM100a / SM103a) and skip elsewhere.
"""

import itertools
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_plugins.models.gdn_chunk_train import ops  # noqa: E402

# A real Qwen3.8-27B TP4 micro-batch: three sequences plus the 512-token alignment pad as its own segment.
LAYOUT = (3339, 2802, 1340, 263)
CHUNK = ops.CHUNK
DEVICE = torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(autouse=True)
def _no_ambient_segment_override(monkeypatch):
    # a leaked override would turn the policy arms into the serial path with every assertion still passing
    monkeypatch.delenv("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS", raising=False)


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
    metas = [ops.chunk_meta_from_host(_cu(LAYOUT), DEVICE) for _ in range(48)]
    assert ops._HOST_META_BUILDS == before + 1, "48 layers of one micro-batch must share one chunk table"
    assert all(m is metas[0] for m in metas)
    starts, lens, seq_chunk_start = _expected_table(LAYOUT)
    assert metas[0].num_chunks == len(starts) and metas[0].num_seqs == len(LAYOUT)
    assert metas[0].chunk_start.tolist() == starts
    assert metas[0].chunk_len.tolist() == lens
    assert metas[0].seq_chunk_start.tolist() == seq_chunk_start
    assert all(t.dtype == torch.int32 and t.is_cuda for t in (metas[0].chunk_start, metas[0].chunk_len))
    # the same layout from a CPU tensor hits the cache; another layout is another build
    assert ops.chunk_meta_from_host(torch.tensor(_cu(LAYOUT), dtype=torch.int64), DEVICE) is metas[0]
    other = ops.chunk_meta_from_host(_cu((8192, 512)), DEVICE)
    assert other is not metas[0] and ops._HOST_META_BUILDS == before + 2


def test_host_boundaries_are_validated():
    with pytest.raises(ValueError):
        ops.chunk_meta_from_host((0, 10, 5), DEVICE)
    with pytest.raises(ValueError):
        ops.chunk_meta_from_host((4,), DEVICE)
    with pytest.raises(ValueError):
        ops.chunk_meta_from_host(torch.tensor([0, 64], device=DEVICE), DEVICE)


def _require_blackwell():
    from miles_plugins.models.gdn_chunk_train import SUPPORTED_CAPABILITIES

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
    dev = ops.build_chunk_meta_from_cu_seqlens(cu, sum(LAYOUT), ops.device_arch(cu.device))
    host = ops.chunk_meta_from_host(_cu(LAYOUT), cu.device)
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
        assert torch.equal(a[name], b[name]), f"{name} differs between the device and host chunk tables at L={host_fwd}"
    monkeypatch.setattr(ops, "FORCED_SEGMENT_LEN", None)
    a = _step(inputs, cu_seqlens=cu)
    b = _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    for name in a:
        assert torch.allclose(a[name].float(), b[name].float(), atol=1e-2, rtol=1e-2), f"{name} differs between the routes' own picks"


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
    per_layer = 9 + 2 * sum(split)  # 4 forward + 5 backward kernels, plus pre-scan + stitch per segmented scan
    assert len(launches) == layers * per_layer, (len(launches), per_layer, sorted(set(runtime)))
    assert not syncs and not copies, sorted(set(runtime))


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
    dev = ops.build_chunk_meta_from_cu_seqlens(cu, sum(LAYOUT), arch, segment_lens=(3, 3))
    seq_chunk_start = _expected_table(LAYOUT)[2]
    for kind in ("fwd_segments", "bwd_segments"):
        h, d = getattr(host, kind), getattr(dev, kind)
        assert h.segment_len == d.segment_len == 3
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
# segment planner (round R2b): the calibrated cost model is a pure function shared with loom's
# ``gdn_chunk_training.plan_segments`` (same constants, same table below), so both tables split identically.
# --------------------------------------------------------------------------- #
B200_SMS, GB300_SMS = 148, 160
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
    assert ops._scan_makespan((133, 4), (3, 1), HV12_CTAS, GB300_SMS, 2) == pytest.approx(45 * ops.SCAN_CO_RESIDENT_FACTOR)
    assert ops._scan_makespan((133, 4), (6, 1), HV12_CTAS, GB300_SMS, 2) == pytest.approx((4 + 23) * ops.SCAN_CO_RESIDENT_FACTOR)
    assert ops._scan_makespan((133, 4), (3, 1), HV12_CTAS, GB300_SMS, 1) == 49.0
    assert ops._scan_makespan((38, 38, 38, 15, 3), (1,) * 5, HV12_CTAS, GB300_SMS, 2) == pytest.approx(38 * ops.SCAN_CO_RESIDENT_FACTOR)


def test_planner_constants_mirror_loom_verbatim():
    # the same literals as loom's gdn_chunk_training.py (its loom/tests/infra/test_gdn_chunk_segment_planner.py pins them)
    assert (ops.MIN_SEGMENT_CHUNKS, ops.SCAN_CO_RESIDENT_FACTOR, ops.PRESCAN_STEP_FACTOR) == (4, 1.60, 1.16)
    assert (ops.PRESCAN_WARMUP_CHUNKS, ops.STITCH_FIXED_CHUNKS, ops.STITCH_COPY_CHUNKS) == (3.3, 2.75, 2.3)
    assert (ops.SPLIT_GAIN_THRESHOLD, ops.MAX_SCAN_ROUNDS) == (0.95, 2)
    assert ops.WARMUP_THRESH_LOG2 == -10.0 / math.log(2.0)


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
    assert ops.plan_segments(136, 8, HV12_CTAS, sm_count, 2)[0] == {B200_SMS: 26, GB300_SMS: 22}[sm_count]
    assert ops.plan_segments(68, 4, HV12_CTAS, sm_count, 2)[0] == 11
    # TP1 (192 CTAs per segment): three segments run two rounds of the resident set (576 CTAs at 3.2x) and still beat
    # the serial scan's 192 co-resident CTAs by 24 % where two CTAs fit one SM; at one CTA per SM nothing beats serial
    assert ops.plan_segments(129, 1, HV48_CTAS, sm_count, 2) == (43, pytest.approx(157.1996))
    assert ops.plan_segments(129, 1, HV48_CTAS, sm_count, 1) == (0, 258.0)
    # documented limitation: a balanced 4 x 1024 layout at four value heads is priced as one long sequence on the bound
    assert ops.plan_segments(68, 4, 4 * (ops.HEAD_DIM // 32), sm_count, 2)[0] == 13


def _split_cost(chunks, seg_len, per_segment, sm_count, residency):
    # the planner's price of segment length ``seg_len`` on ``chunks``: pass-2 makespan over every sequence's segments,
    # warm-up pre-scan of the consumed segments' value blocks, stitch launch floor + one copy-only step per boundary
    splits = tuple(max(1, -(-n // seg_len)) for n in chunks)
    return (
        ops._scan_makespan(chunks, splits, per_segment, sm_count, residency)
        + ops.PRESCAN_WARMUP_CHUNKS * ops.PRESCAN_STEP_FACTOR * ops._wave_factor(sum(k - 1 for k in splits) * per_segment, sm_count, residency)
        + ops.STITCH_FIXED_CHUNKS
        + ops.STITCH_COPY_CHUNKS * (max(splits) - 1)
    )


def _admissible_segment_lens(chunks, per_segment, sm_count, residency):
    longest = max(chunks)
    per_wave = sm_count * (2 if residency >= 2 else 1)
    return {
        -(-longest // s)
        for s in range(2, longest // ops.MIN_SEGMENT_CHUNKS + 1)
        if ops.MIN_SEGMENT_CHUNKS <= -(-longest // s) < longest
        and sum(max(1, -(-n // -(-longest // s))) for n in chunks) * per_segment <= ops.MAX_SCAN_ROUNDS * per_wave
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
            admissible = {length: _split_cost(chunks, length, per_segment, sm_count, residency) for length in _admissible_segment_lens(chunks, per_segment, sm_count, residency)}
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
    assert ops.plan_segments(138, 2, HV12_CTAS, sm_count, 2, seq_chunks=chunks(8481, 223)) == (27, pytest.approx(61.2748))
    assert ops.plan_segments(138, 2, HV12_CTAS, sm_count, 1, seq_chunks=chunks(8481, 223)) == (45, pytest.approx(60.178))
    assert ops.segment_bounds(133, 27) == [0, 26, 53, 79, 106, 133]
    assert ops.segment_bounds(133, 45) == ops.segment_bounds(133, 46) == [0, 44, 88, 133]
    assert ops.plan_segments(260, 2, HV12_CTAS, sm_count, 2, seq_chunks=chunks(16384, 512))[0] == 52
    assert ops.plan_segments(260, 2, HV12_CTAS, sm_count, 1, seq_chunks=chunks(16384, 512))[0] == 86
    # two long sequences plus the pseudo-sequence (69 + 55 + 6 chunks): the exact layout splits fwd_h at L = 23
    # (3 + 3 + 1 segments, 59.9 vs 69 chunk steps) and keeps dhu serial (one CTA per SM); the bound splits 22 / 43
    assert ops.plan_segments(131, 3, HV12_CTAS, sm_count, 2)[0] == 22
    assert ops.plan_segments(131, 3, HV12_CTAS, sm_count, 1)[0] == 43
    assert ops.plan_segments(131, 3, HV12_CTAS, sm_count, 2, seq_chunks=chunks(4360, 3491, 341)) == (23, pytest.approx(59.8748))
    assert ops.plan_segments(131, 3, HV12_CTAS, sm_count, 1, seq_chunks=chunks(4360, 3491, 341)) == (0, 69.0)
    # packs: 44 / 44 / 39 / 3 chunks (192 CTAs, co-resident even unsplit) split fwd_h at 22; the real five-sequence
    # micro-batch (38/38/38/15/3 chunks, 240 CTAs) stays serial on both scans (the R3 pre-fit planner split it at
    # L = 19 and measured +5 % on the forward); 4 x 4096 splits fwd_h at L = 22 (12 segments, two rounds at 3.2x: 90
    # vs 102.4 chunk steps for 192 co-resident CTAs) and keeps the one-CTA-per-SM dhu scan serial; 4 x 1024 at HV 4
    # stays serial
    assert ops.plan_segments(132, 4, HV12_CTAS, sm_count, 2, seq_chunks=chunks(2802, 2771, 2476, 143)) == (22, pytest.approx(48.878))
    assert ops.plan_segments(133, 5, HV12_CTAS, sm_count, 2, seq_chunks=chunks(2384, 2382, 2369, 902, 155)) == (0, pytest.approx(60.8))
    assert ops.plan_segments(260, 4, HV12_CTAS, sm_count, 2, seq_chunks=chunks(4096, 4096, 4096, 4096)) == (22, pytest.approx(89.9996))
    assert ops.plan_segments(260, 4, HV12_CTAS, sm_count, 1, seq_chunks=chunks(4096, 4096, 4096, 4096)) == (0, 128.0)
    assert ops.plan_segments(130, 2, 24 * (ops.HEAD_DIM // 32), sm_count, 2, seq_chunks=chunks(4096, 4096)) == (22, pytest.approx(89.9996))
    assert ops.plan_segments(68, 4, 16, sm_count, 2, seq_chunks=chunks(1024, 1024, 1024, 1024)) == (0, 16.0)
    # the wiring LAYOUT at HV 4 splits both scans at 18 on the exact layout
    assert ops.plan_segments(125, 4, 16, sm_count, 2, seq_chunks=chunks(3339, 2802, 1340, 263)) == (18, pytest.approx(29.178))


def test_host_route_prices_the_exact_layout():
    _require_blackwell()
    ops._host_chunk_meta.cache_clear()
    arch = ops.device_arch(DEVICE)
    index = DEVICE.index
    # two long sequences + the 512-alignment pseudo-sequence: the device route (chunk bound only) prices one 129-chunk
    # chain (22 / 43), the host route the exact 69 + 55 + 6 layout (fwd_h 23: 3 + 3 + 1 segments; dhu serial)
    pack = (4360, 3491, 341)
    assert ops.segment_lens(sum(pack), 3, 12, index, arch) == (22, 43)
    assert ops.segment_lens(sum(pack), 3, 12, index, arch, pack) == (23, 0)
    meta = ops.chunk_meta_from_host(_cu(pack), DEVICE, 12, arch=arch)
    assert meta.fwd_segments.segment_len == 23 and meta.bwd_segments.segment_len == 0
    assert meta.fwd_segments.num_segments == 7 and meta.bwd_segments.num_segments == len(pack)
    assert meta.fwd_segments.seg_chunk_start.tolist() == [0, 23, 46, 69, 87, 105, 124, 130]
    # one dominant sequence + pad: the device route (bound 137 + 1) picks 23 / 46, the exact layout 27 (five fwd_h
    # segments: the pad's CTAs make every grid above one CTA per SM co-resident) / 45 (three dhu segments)
    mb00 = (8481, 223)
    assert ops.segment_lens(sum(mb00), 2, 12, index, arch) == (23, 46)
    assert ops.segment_lens(sum(mb00), 2, 12, index, arch, mb00) == (27, 45)
    host = ops.chunk_meta_from_host(_cu(mb00), DEVICE, 12, arch=arch)
    assert host.fwd_segments.segment_len == 27 and host.fwd_segments.num_segments == 6  # 5 segments + the pad sequence
    assert host.fwd_segments.seg_chunk_start.tolist() == [0, 26, 53, 79, 106, 133, 137]
    assert host.bwd_segments.segment_len == 45 and host.bwd_segments.seg_chunk_start.tolist() == [0, 44, 88, 133, 137]


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
    with pytest.warns(RuntimeWarning, match=r"forced segment length 43 ignored for the dhu scan .*\(64, 1\)"):
        assert ops.segment_lens(16384, 5, 12, DEVICE.index, arch) == (43, 0)
