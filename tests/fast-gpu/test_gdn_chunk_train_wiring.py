"""Host wiring of the deterministic chunked GDN kernels (``miles_plugins/models/gdn_chunk_train``).

One exact chunk table per micro-batch layout (shared by every layer, no device-to-host synchronisation), launch
counts, and the bit-identical tape / recompute backward.  Needs a CUDA device; the kernel tests need Blackwell
(SM100a / SM103a) and skip elsewhere.
"""

import itertools

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_plugins.models.gdn_chunk_train import ops  # noqa: E402

# A real Qwen3.8-27B TP4 micro-batch: three sequences plus the 512-token alignment pad as its own segment.
LAYOUT = (3339, 2802, 1340, 263)
CHUNK = ops.CHUNK
DEVICE = torch.device("cuda", torch.cuda.current_device())


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


def test_host_and_device_tables_give_identical_results():
    _require_blackwell()
    inputs = _inputs(LAYOUT)
    cu = torch.tensor(_cu(LAYOUT), device="cuda", dtype=torch.int32)
    a = _step(inputs, cu_seqlens=cu)
    b = _step(inputs, cu_seqlens=cu, cu_seqlens_cpu=_cu(LAYOUT))
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs between the device and host chunk tables"


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


def test_wiring_has_no_host_sync_and_nine_launches_per_layer(monkeypatch):
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
    assert len(launches) == layers * 9, (len(launches), sorted(set(runtime)))  # 4 forward + 5 backward kernels
    assert not syncs and not copies, sorted(set(runtime))
