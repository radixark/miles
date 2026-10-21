"""Boundary contract of the KDA layers: device int32 ``cu_seqlens`` + one CPU int64 ``cu_seqlens_cpu`` per
micro-batch, reused by every KDA layer and both backends, with no device synchronisation on the hit path.

Routing / contract checks run on any machine with torch (and Megatron for the layer-level ones). The GPU
checks need a Blackwell (SM100a/SM103a) device with flash-linear-attention and the built deterministic
backward; they skip otherwise.

Sync-free check (``test_sync_free_after_one_warmup``): ``torch.cuda.set_sync_debug_mode("error")`` raises
on the CUDA calls torch's ``c10::cuda::warn_or_error_on_sync`` guards. Confirmed on the compute node
(nvl72d377-T09, GB300, torch 2.13.0+cu130, ``sync_debug_probe.py`` in the step log of lease 919049 step
s10): FLAGGED -- ``.item()`` / ``.tolist()`` / ``.cpu()`` of a CUDA tensor, a blocking ``copy_`` in either
direction (``torch.tensor(list, device="cuda")``, ``.to("cuda")`` from pageable memory: memcpy + stream
drain), ``Stream.synchronize()``, and the data-dependent-shape ops ``nonzero`` / ``masked_select`` /
``repeat_interleave`` with tensor repeats / ``bool(tensor)``. NOT flagged -- ``copy_(non_blocking=True)``
from pinned or pageable memory, ``.to("cpu", non_blocking=True)``, ``pin_memory`` allocations, kernel
launches, ``cumsum``; and, on this build, also ``torch.cuda.synchronize()``, ``Event.synchronize()`` and
``Event.query()`` (torch itself warns that the mode "does not yet detect all synchronizing operations").
Explicit stream / device / event synchronisation inside the op is therefore not covered by this test; the
U0-1 CUPTI count of ``cuda*Synchronize`` records in the KDA windows covers it.
"""

from __future__ import annotations

import time
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles_plugins.models.kda_chunk_train import backward as det_backward
from miles_plugins.models.kda_chunk_train import kda_backend as drop_in

CHUNK = det_backward.CHUNK

# packings the layout tables are checked on: RL packs of the production micro-batch, the padded 8-sequence
# mix with its tail pad pseudo-sequence, 1-token sequences, chunk-aligned and identity layouts
PACKS = {
    "packed_rl3_8192": (2300, 3100, 2792),
    "packed_rl5_8192": (1200, 2500, 900, 1980, 1612),
    "packed_8mix_8192_tailpad": (1, 63, 65, 127, 1000, 1500, 2048, 3000, 388),
    "one_token": (1,),
    "one_token_runs": (1, 1, 1, 61),
    "one_token_between": (64, 1, 63, 1),
    "aligned_identity": (128, 64, 192),
    "aligned_odd_chunks": (64, 128),
}


def _cu(lengths):
    offs = [0]
    for length in lengths:
        offs.append(offs[-1] + int(length))
    return offs


# ------------------------------------------------------------------------------------- reference tables
def _reference_tables(lengths, offsets, rows_external):
    """The PR-head Python loop (``_build_layout`` before vectorisation): the tables the kernels were validated on."""
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + -(-length // CHUNK))
    nc = starts[-1]
    nc_pad = -(-nc // 2) * 2
    bos = [0] * nc_pad
    clen = [0] * nc_pad
    internal = [None] * rows_external
    for n, (length, offset) in enumerate(zip(lengths, offsets, strict=True)):
        c0 = starts[n]
        for i in range(starts[n + 1] - c0):
            bos[c0 + i] = offset + i * CHUNK
            clen[c0 + i] = min(CHUNK, length - i * CHUNK)
        for j in range(length):
            internal[offset + j] = c0 * CHUNK + j
    return starts, bos, clen, internal


def _reference_chunk_indices(starts):
    """fla's ``prepare_chunk_indices`` rows: (sequence, chunk index within it) per real chunk."""
    return [[n, i] for n in range(len(starts) - 1) for i in range(starts[n + 1] - starts[n])]


def _tables(layout):
    return [
        list(layout.seq_chunk_start_cpu),
        layout.chunk_bos.tolist(),
        layout.chunk_len.tolist(),
        layout.internal_rows.tolist(),
    ]


@pytest.mark.parametrize("name", sorted(PACKS))
def test_host_tables_match_the_reference_tables(name):
    """Vectorised host tables == the reference loop's tables, and the device-side tables carry them."""
    lengths = PACKS[name]
    cu = _cu(lengths)
    total = cu[-1]
    cu_dev = torch.tensor(cu, dtype=torch.int32)
    det_backward._LAYOUT_CACHE.clear()
    layout = det_backward.chunk_layout(1, total, cu_dev, torch.device("cpu"), cu_seqlens_cpu=tuple(cu))
    assert "internal_rows" not in vars(layout)  # the per-row scatter table is built on first use only
    starts, bos, clen, internal = _reference_tables(lengths, cu[:-1], total)
    got_starts, got_bos, got_clen, got_internal = _tables(layout)
    assert got_starts == starts
    assert got_bos == bos
    assert got_clen == clen
    assert got_internal == internal
    assert layout.seq_chunk_start.tolist() == starts
    assert layout.chunk_bos.dtype == layout.chunk_len.dtype == layout.seq_chunk_start.dtype == torch.int32
    assert layout.internal_rows.dtype == torch.int64
    assert "internal_rows" in vars(layout) and layout.internal_rows is layout.internal_rows
    assert layout.chunk_indices.dtype == torch.int32 and tuple(layout.chunk_indices.shape) == (starts[-1], 2)
    assert layout.chunk_indices.tolist() == _reference_chunk_indices(starts)
    assert layout.cp_cu_first.dtype == layout.cp_cu_last.dtype == torch.int32
    assert layout.cp_cu_first.tolist() == [0, lengths[0]] and layout.cp_cu_last.tolist() == [0, lengths[-1]]
    assert layout.num_chunks == starts[-1] and layout.num_chunks_padded % 2 == 0
    assert layout.identity == (name == "aligned_identity")


def test_internal_rows_is_built_on_first_scatter_only():
    """The int64 per-row table is the only layout table a forward never needs: built (and uploaded) by the first
    ``scatter``; an identity layout never builds it."""
    det_backward._LAYOUT_CACHE.clear()
    cu = _cu(PACKS["packed_rl3_8192"])
    layout = det_backward.chunk_layout(1, cu[-1], torch.tensor(cu, dtype=torch.int32), "cpu", cu_seqlens_cpu=tuple(cu))
    assert "internal_rows" not in vars(layout)
    x = torch.arange(layout.rows, dtype=torch.float32).unsqueeze(1)
    got = layout.scatter(x)
    assert "internal_rows" in vars(layout)
    assert got.squeeze(1).tolist() == [float(i) for i in _reference_tables(layout.lengths, layout.offsets, cu[-1])[3]]
    ident = det_backward.chunk_layout(1, 384, torch.tensor([0, 128, 192, 384], dtype=torch.int32), "cpu", cu_seqlens_cpu=(0, 128, 192, 384))
    y = x[: ident.rows]
    assert ident.identity and ident.scatter(y) is y and "internal_rows" not in vars(ident)


@pytest.mark.parametrize("batch, seq_len", [(1, 100), (2, 128), (3, 1000)])
def test_fixed_length_tables_match_the_reference_tables(batch, seq_len):
    det_backward._LAYOUT_CACHE.clear()
    layout = det_backward.chunk_layout(batch, seq_len, None, torch.device("cpu"))
    lengths, offsets = [seq_len] * batch, [b * seq_len for b in range(batch)]
    assert _tables(layout) == [list(t) for t in _reference_tables(lengths, offsets, batch * seq_len)]


def test_tensor_tuple_and_device_read_share_one_layout_entry():
    """The cache key is the host tuple: an int64 CPU tensor, a tuple and the ``tolist()`` fallback hit one entry."""
    lengths = PACKS["packed_rl5_8192"]
    cu = _cu(lengths)
    cu_dev = torch.tensor(cu, dtype=torch.int32)
    det_backward._LAYOUT_CACHE.clear()
    host = torch.tensor(cu, dtype=torch.int64)
    from_tensor = det_backward.chunk_layout(1, cu[-1], cu_dev, "cpu", cu_seqlens_cpu=host)
    from_tuple = det_backward.chunk_layout(1, cu[-1], cu_dev, "cpu", cu_seqlens_cpu=tuple(cu))
    from_device = det_backward.chunk_layout(1, cu[-1], cu_dev, "cpu")  # tolist() fallback, non-default
    assert from_tensor is from_tuple is from_device
    assert len(det_backward._LAYOUT_CACHE) == 1
    assert _tables(from_tensor) == [list(t) for t in _reference_tables(lengths, cu[:-1], cu[-1])]


def test_layout_cache_is_an_lru():
    det_backward._LAYOUT_CACHE.clear()
    limit = det_backward._LAYOUT_CACHE_LIMIT
    def layout(total):
        cu = torch.tensor([0, total], dtype=torch.int32)
        return det_backward.chunk_layout(1, total, cu, "cpu", cu_seqlens_cpu=(0, total))

    first = layout(64)
    for n in range(1, limit + 8):
        layout(64 + n)
        # touching the first entry keeps it resident while the oldest untouched entries are evicted
        assert layout(64) is first
    assert len(det_backward._LAYOUT_CACHE) == limit


def test_host_boundaries_accepts_host_tensors_and_sequences_only():
    assert det_backward.host_boundaries(torch.tensor([0, 3, 7], dtype=torch.int64)) == (0, 3, 7)
    assert det_backward.host_boundaries(torch.tensor([0, 3, 7], dtype=torch.int32)) == (0, 3, 7)
    assert det_backward.host_boundaries([0, 3, 7]) == (0, 3, 7)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="host"):
            det_backward.host_boundaries(torch.tensor([0, 3, 7], device="cuda"))


# ------------------------------------------------------------------------------------- wiring (CPU)
class _Recorder:
    """Stands in for a KDA kernel: records the arguments of every call (fla's ``chunk_kda`` takes q/k/v/g/beta
    positionally on the fallback path, ``kda_recurrence`` passes everything by keyword)."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return (args[0] if args else kwargs["q"]), None


def _small_operands(seq_len=8, heads=2, device="cpu"):
    q = torch.randn(1, seq_len, heads, 128, device=device, dtype=torch.bfloat16)
    return dict(
        q=q,
        k=q.clone(),
        v=q.clone(),
        beta_logits=torch.randn(1, seq_len, heads, device=device, dtype=torch.bfloat16),
        decay=torch.randn(1, seq_len, heads * 128, device=device, dtype=torch.bfloat16),
        A_log=torch.zeros(heads, device=device),
        dt_bias=torch.zeros(heads * 128, device=device),
    )


@pytest.mark.parametrize("backend", ["fla", "deterministic"])
def test_kda_recurrence_hands_the_host_copy_and_state_v_first_to_both_backends(monkeypatch, backend):
    pytest.importorskip("megatron.core")
    from miles_plugins.models import linear_attn

    recorder = _Recorder()
    monkeypatch.setattr(linear_attn, "kda_kernel", lambda name: recorder)
    t = _small_operands()
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    host = torch.tensor([0, 3, 8], dtype=torch.int64)
    linear_attn.kda_recurrence(
        t["q"], t["k"], t["v"], t["beta_logits"], t["decay"], t["A_log"], t["dt_bias"],
        gate_lower_bound=-5.0, cu_seqlens=cu, cp_context=None, backend=backend, cu_seqlens_cpu=host,
    )
    (call,) = recorder.calls
    assert call["cu_seqlens"] is cu and call["cu_seqlens_cpu"] is host
    assert call["state_v_first"] is True and "transpose_state_layout" not in call
    # under CP the kernel takes the boundaries from the context; the host copy still travels along
    ctx = object()
    linear_attn.kda_recurrence(
        t["q"], t["k"], t["v"], t["beta_logits"], t["decay"], t["A_log"], t["dt_bias"],
        gate_lower_bound=-5.0, cu_seqlens=None, cp_context=ctx, backend=backend, cu_seqlens_cpu=host,
    )
    call = recorder.calls[-1]
    assert call["cp_context"] is ctx and "cu_seqlens" not in call and call["cu_seqlens_cpu"] is host


def test_packed_seq_params_carry_one_int64_host_tensor_per_microbatch():
    pytest.importorskip("megatron.training")
    from miles.backends.megatron_utils.parallel import PackedSeqParamsWithHostCuSeqlens, get_packed_seq_params

    host_tuple = (0, 1200, 3700, 4600, 6580, 8192)
    cu = torch.tensor(host_tuple, dtype=torch.int32)
    batch = {"cu_seqlens": cu, "cu_seqlens_host": host_tuple, "max_seqlen": 2500}
    params = get_packed_seq_params(batch, SimpleNamespace(qkv_format="thd"))
    assert isinstance(params, PackedSeqParamsWithHostCuSeqlens) and batch["packed_seq_params"] is params
    assert params.cu_seqlens_cpu.dtype == torch.int64 and params.cu_seqlens_cpu.device.type == "cpu"
    assert tuple(params.cu_seqlens_cpu.tolist()) == host_tuple == params.cu_seqlens_host
    assert params.cu_seqlens_q is batch["cu_seqlens"]
    assert params.fla_cp_context is None and params.fla_cp_cu_seqlens_cpu is None
    # the object is built once; the host tensor is one object for the micro-batch
    assert params.cu_seqlens_cpu is params.cu_seqlens_cpu
    assert get_packed_seq_params(dict(batch), SimpleNamespace(qkv_format="bshd")) is None


class _Group:
    """Process-group stand-in (size / rank only; the collectives are patched to identities)."""

    def __init__(self, size, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _KDAStub(nn.Module):
    """``LinearAttention`` stand-in: records what the layer hands to the kernel path and returns its input."""

    conv_kernel_size = 4

    def __init__(self):
        super().__init__()
        self.calls = []
        self.out_proj = nn.Identity()

    def forward(self, x, cu_seqlens, cp_context=None, cu_seqlens_cpu=None):
        self.calls.append(SimpleNamespace(cu_seqlens=cu_seqlens, cp_context=cp_context, cu_seqlens_cpu=cu_seqlens_cpu))
        return x


def _layers(monkeypatch, n, cp_size):
    pytest.importorskip("megatron.core")
    from miles_plugins.models import linear_attn

    for name in ("copy_to_tensor_model_parallel_region", "reduce_from_tensor_model_parallel_region"):
        monkeypatch.setattr(linear_attn, name, lambda x, group=None: x)
    pg = SimpleNamespace(tp=_Group(1), cp=_Group(cp_size, rank=0))
    config = SimpleNamespace(sequence_parallel=False)
    return [
        linear_attn.LinearAttentionLayer(config, _KDAStub(), nn.Identity(), pg, allgather_cp=True) for _ in range(n)
    ]


def _params(host_tuple):
    from miles.backends.megatron_utils.parallel import get_packed_seq_params

    batch = {"cu_seqlens": torch.tensor(host_tuple, dtype=torch.int32), "cu_seqlens_host": host_tuple, "max_seqlen": 0}
    return get_packed_seq_params(batch, SimpleNamespace(qkv_format="thd"))


def test_every_layer_of_a_microbatch_receives_the_same_host_object(monkeypatch):
    """9 forwards of 3 layers on one ``PackedSeqParams`` (first forward, recompute forward, ...): one ``id()``."""
    pytest.importorskip("megatron.training")
    layers = _layers(monkeypatch, 3, cp_size=1)
    params = _params((0, 5, 12))
    x = torch.zeros(12, 1, 8)
    for _ in range(3):
        for layer in layers:
            layer(x, packed_seq_params=params)
    seen = [call for layer in layers for call in layer.linear_attn.calls]
    assert len(seen) == 9
    assert {id(call.cu_seqlens_cpu) for call in seen} == {id(params.cu_seqlens_cpu)}
    assert all(call.cu_seqlens is params.cu_seqlens_q and call.cp_context is None for call in seen)
    assert params.cu_seqlens_cpu.dtype == torch.int64
    # the next micro-batch brings its own object
    params2 = _params((0, 5, 12))
    layers[0](x, packed_seq_params=params2)
    assert layers[0].linear_attn.calls[-1].cu_seqlens_cpu is params2.cu_seqlens_cpu is not params.cu_seqlens_cpu


def test_without_a_host_copy_the_layer_passes_none(monkeypatch):
    """A foreign ``PackedSeqParams`` / bshd input: no host copy -> ``None`` (the backends fall back / self-serve)."""
    pytest.importorskip("megatron.core")
    from megatron.core.packed_seq_params import PackedSeqParams

    (layer,) = _layers(monkeypatch, 1, cp_size=1)
    x = torch.zeros(12, 1, 8)
    foreign = PackedSeqParams(qkv_format="thd", cu_seqlens_q=torch.tensor([0, 12], dtype=torch.int32))
    layer(x, packed_seq_params=foreign)
    layer(x, packed_seq_params=None)
    assert [c.cu_seqlens_cpu for c in layer.linear_attn.calls] == [None, None]
    assert layer.linear_attn.calls[0].cu_seqlens is foreign.cu_seqlens_q
    assert layer.linear_attn.calls[1].cu_seqlens.tolist() == [0, 12]


def test_cp_context_is_built_once_per_microbatch_and_shared_by_every_layer(monkeypatch):
    """Under CP the first layer builds the fla context with the micro-batch's host copy; the other layers and the
    recompute forwards reuse that one object (and one int64 host copy of the rank-local boundaries)."""
    pytest.importorskip("megatron.training")
    from miles_plugins.models import linear_attn

    built = []

    def fake_build(cu_seqlens, cp_group, conv_kernel_size, device, cu_seqlens_cpu=None):
        local = torch.tensor([0, 6], dtype=torch.int32)
        ctx = SimpleNamespace(
            group=cp_group, conv1d_kernel_size=conv_kernel_size, cu_seqlens=local, cu_seqlens_cpu=local
        )
        built.append(SimpleNamespace(ctx=ctx, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu))
        return ctx

    monkeypatch.setattr(linear_attn, "build_fla_cp_context", fake_build)
    layers = _layers(monkeypatch, 3, cp_size=2)
    params = _params((0, 5, 12))
    x = torch.zeros(6, 1, 8)
    for _ in range(3):
        for layer in layers:
            layer(x, packed_seq_params=params)
    seen = [call for layer in layers for call in layer.linear_attn.calls]
    assert len(built) == 1 and len(seen) == 9
    assert built[0].cu_seqlens_cpu is params.cu_seqlens_cpu and built[0].cu_seqlens is params.cu_seqlens_q
    assert {id(call.cp_context) for call in seen} == {id(built[0].ctx)} == {id(params.fla_cp_context)}
    assert {id(call.cu_seqlens_cpu) for call in seen} == {id(params.fla_cp_cu_seqlens_cpu)}
    assert params.fla_cp_cu_seqlens_cpu.dtype == torch.int64 and params.fla_cp_cu_seqlens_cpu.tolist() == [0, 6]
    assert all(call.cu_seqlens is built[0].ctx.cu_seqlens for call in seen)
    # a new micro-batch object builds its own context; a foreign PackedSeqParams builds per call (no host copy)
    layers[0](x, packed_seq_params=_params((0, 5, 12)))
    assert len(built) == 2
    from megatron.core.packed_seq_params import PackedSeqParams

    foreign = PackedSeqParams(qkv_format="thd", cu_seqlens_q=torch.tensor([0, 12], dtype=torch.int32))
    layers[0](x, packed_seq_params=foreign)
    layers[0](x, packed_seq_params=foreign)
    assert len(built) == 4 and built[-1].cu_seqlens_cpu is None


# ------------------------------------------------------------------------------------- drop-in routing (CPU)
def _drop_in_kwargs(t, **extra):
    return dict(
        q=t["q"], k=t["k"], v=t["v"], g=t["decay"].reshape(t["v"].shape), beta=t["beta_logits"].float().sigmoid(),
        scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, state_v_first=True,
        A_log=t["A_log"], dt_bias=t["dt_bias"], **extra,
    )


@pytest.mark.parametrize(
    ("forward_source", "expected_source"),
    [(None, "cake"), ("cake", "cake"), ("fla", "fla")],
    ids=["default-alias", "cake", "fla"],
)
def test_cp_routes_to_the_deterministic_path_on_the_context_boundaries_with_no_warning(
    monkeypatch, forward_source, expected_source
):
    """Context parallelism is in the domain. With an fla CP context the drop-in takes
    the rank-local boundaries from the context (device ``cu_seqlens`` and its host copy, as fla overrides them),
    hands the context and the forward source to the deterministic autograd function and does not warn; fla's
    ``chunk_kda`` is not called. The default alias is the Cake forward, whose contract takes beta logits with
    ``use_beta_sigmoid_in_kernel=True``; ``forward_source="fla"`` keeps fla's convention (post-sigmoid beta).
    (``build_fla_cp_context`` receiving the micro-batch's host copy: ``test_cp_context_is_built_once_per_microbatch``.)"""
    fla_kda = pytest.importorskip("fla.ops.kda")
    fla_recorder = _Recorder()
    monkeypatch.setattr(fla_kda, "chunk_kda", fla_recorder)
    monkeypatch.setattr(drop_in, "_fallback_warned", False)
    monkeypatch.setattr(drop_in, "_BLACKWELL_CAPABILITIES", ((0, 0),))  # CPU operands report capability (0, 0)
    applied = []

    def fake_apply(*args):
        applied.append(args)
        return args[0]

    monkeypatch.setattr(drop_in, "_ChunkKDADeterministicBackward", SimpleNamespace(apply=fake_apply))
    t = _small_operands()
    local = torch.tensor([0, 8], dtype=torch.int32)
    host = torch.tensor([0, 8], dtype=torch.int64)
    ctx = SimpleNamespace(group=object(), cu_seqlens=local, cu_seqlens_cpu=host, is_first_rank=True, is_last_rank=False)
    kwargs = _drop_in_kwargs(t, cu_seqlens_cpu=host, cp_context=ctx)
    if expected_source == "cake":
        kwargs.update(beta=t["beta_logits"], use_beta_sigmoid_in_kernel=True)
    if forward_source is not None:
        kwargs["forward_source"] = forward_source
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            out, _ = drop_in.chunk_kda(**kwargs)
    assert not [w for w in caught if "fell back to fla" in str(w.message)]
    assert not fla_recorder.calls and len(applied) == 3 and out is t["q"]
    for args in applied:
        beta, cu_seqlens, cu_seqlens_cpu, source, cp_context = args[4], args[9], args[10], args[11], args[12]
        assert beta is kwargs["beta"] and cu_seqlens is local and cu_seqlens_cpu is host and cp_context is ctx
        assert source == expected_source


def test_cp_context_without_a_host_copy_falls_back_to_fla_with_exactly_one_warning(monkeypatch):
    """A context object carrying no host copy of the local boundaries (fla's ``build_cp_context`` always fills
    ``cu_seqlens_cpu``, so this is the drop-in's rule for any other context object) routes to fla with the caller's
    arguments untouched, once-warned, like packed input without a host copy."""
    fla_kda = pytest.importorskip("fla.ops.kda")
    recorder = _Recorder()
    monkeypatch.setattr(fla_kda, "chunk_kda", recorder)
    monkeypatch.setattr(drop_in, "_fallback_warned", False)
    monkeypatch.setattr(drop_in, "_BLACKWELL_CAPABILITIES", ((0, 0),))
    t = _small_operands()
    ctx = SimpleNamespace(group=object(), cu_seqlens=torch.tensor([0, 8], dtype=torch.int32), cu_seqlens_cpu=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            out, _ = drop_in.chunk_kda(**_drop_in_kwargs(t, cu_seqlens_cpu=None, cp_context=ctx))
    fallbacks = [w for w in caught if "fell back to fla" in str(w.message)]
    assert len(fallbacks) == 1 and "cu_seqlens_cpu" in str(fallbacks[0].message)
    assert len(recorder.calls) == 2 and out is t["q"]
    for call in recorder.calls:
        assert call["cp_context"] is ctx and call["cu_seqlens"] is None and call["cu_seqlens_cpu"] is None


def test_packed_input_without_a_host_copy_falls_back_to_fla_with_exactly_one_warning(monkeypatch):
    fla_kda = pytest.importorskip("fla.ops.kda")
    recorder = _Recorder()
    monkeypatch.setattr(fla_kda, "chunk_kda", recorder)
    monkeypatch.setattr(drop_in, "_fallback_warned", False)
    t = _small_operands()
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            drop_in.chunk_kda(**_drop_in_kwargs(t, cu_seqlens=cu, cu_seqlens_cpu=None))
    fallbacks = [w for w in caught if "fell back to fla" in str(w.message)]
    assert len(fallbacks) == 1 and "cu_seqlens_cpu" in str(fallbacks[0].message)
    assert len(recorder.calls) == 2
    assert all(c["cu_seqlens"] is cu and c["cu_seqlens_cpu"] is None for c in recorder.calls)


def test_device_capability_is_cached_per_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: calls.append(device) or (10, 3))
    drop_in._device_capability.cache_clear()
    for _ in range(5):
        assert drop_in._device_capability(torch.device("cuda", 0)) == (10, 3)
    assert drop_in._device_capability(torch.device("cuda", 1)) == (10, 3)
    assert calls == [torch.device("cuda", 0), torch.device("cuda", 1)]
    drop_in._device_capability.cache_clear()


# ------------------------------------------------------------------------------------- GPU
_HEADS = 4
_GRADS = ("q", "k", "v", "decay", "beta_logits", "A_log", "dt_bias")


def _kda_recurrence():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the deterministic KDA backward requires SM100a or SM103a")
    pytest.importorskip("fla.ops.kda")
    pytest.importorskip("megatron.core")
    from miles_plugins.models.linear_attn import kda_recurrence

    return kda_recurrence


def _inputs(total, seed, heads=_HEADS):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def act():
        return (torch.randn(1, total, heads, 128, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)

    return {
        "q": act(),
        "k": act(),
        "v": act(),
        "decay": act().reshape(1, total, heads * 128),
        "beta_logits": torch.randn(1, total, heads, generator=gen, device="cuda").to(torch.bfloat16),
        "A_log": torch.log(torch.rand(heads, generator=gen, device="cuda") + 1.0),
        "dt_bias": torch.randn(heads * 128, generator=gen, device="cuda"),
    }


def _fwd_bwd(kda_recurrence, backend, inputs, cu, host, upstream):
    leaves = {k: v.detach().requires_grad_(True) for k, v in inputs.items()}
    out = kda_recurrence(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["beta_logits"],
        leaves["decay"],
        leaves["A_log"],
        leaves["dt_bias"],
        gate_lower_bound=-5.0,
        cu_seqlens=cu,
        cp_context=None,
        backend=backend,
        cu_seqlens_cpu=host,
    )
    grads = torch.autograd.grad(out, [leaves[k] for k in _GRADS], grad_outputs=upstream)
    return out.detach(), dict(zip(_GRADS, [g.detach() for g in grads], strict=True))


def _fwd_only(kda_recurrence, backend, inputs, cu, host):
    with torch.no_grad():
        return kda_recurrence(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["beta_logits"],
            inputs["decay"],
            inputs["A_log"],
            inputs["dt_bias"],
            gate_lower_bound=-5.0,
            cu_seqlens=cu,
            cp_context=None,
            backend=backend,
            cu_seqlens_cpu=host,
        )


def test_sync_free_after_one_warmup():
    """U0-2: after one warm-up on the pack, forward + backward of both backends run with
    ``set_sync_debug_mode("error")`` (see the module docstring for the ops it flags). The deterministic
    backend also stays sync-free for a new boundary object of the same pack (its caches key on the host
    tuple); fla's identity-keyed caches would re-upload the chunk indices for a new object, which U0-1(c)
    allows once per micro-batch and which this check therefore leaves out."""
    kda_recurrence = _kda_recurrence()
    lengths = PACKS["packed_rl5_8192"]
    cu_list = _cu(lengths)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    host = torch.tensor(cu_list, dtype=torch.int64)
    inputs = _inputs(cu_list[-1], seed=579)
    upstream = torch.randn_like(inputs["v"])
    # the next micro-batch's boundary objects (same pack): inputs of the op, built outside the checked region
    cu2 = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    host2 = torch.tensor(cu_list, dtype=torch.int64)
    for backend in ("fla", "deterministic"):
        _fwd_bwd(kda_recurrence, backend, inputs, cu, host, upstream)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for backend in ("fla", "deterministic"):
            _fwd_bwd(kda_recurrence, backend, inputs, cu, host, upstream)
        _fwd_bwd(kda_recurrence, "deterministic", inputs, cu2, host2, upstream)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_new_pack_layout_build_does_not_block_the_stream():
    """Deliverable 3: the first layout of a never-seen packing uploads its tables without a stream drain."""
    _kda_recurrence()
    det_backward._LAYOUT_CACHE.clear()
    torch.cuda.synchronize()
    device = torch.device("cuda", torch.cuda.current_device())
    timings = []
    torch.cuda.set_sync_debug_mode("error")
    try:
        for n in range(24):
            lengths = (1200 + n, 2500 - n, 900, 1980, 1612)
            cu_list = _cu(lengths)
            cu = torch.tensor(cu_list, dtype=torch.int32).to(device, non_blocking=True)  # pageable, no drain
            t0 = time.perf_counter()
            layout = det_backward.chunk_layout(1, cu_list[-1], cu, device, cu_seqlens_cpu=tuple(cu_list))
            timings.append(time.perf_counter() - t0)
            assert layout.rows_external == cu_list[-1]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    layout = det_backward.chunk_layout(1, cu_list[-1], cu, device, cu_seqlens_cpu=tuple(cu_list))
    assert _tables(layout) == [list(t) for t in _reference_tables(lengths, cu_list[:-1], cu_list[-1])]
    median_us, max_us = sorted(timings)[12] * 1e6, max(timings) * 1e6
    print(f"\n[chunk_layout host build, 24 new packs] median {median_us:.0f} us max {max_us:.0f} us")


def test_tensor_and_tuple_host_copies_give_identical_results():
    """Kernel-level direct calls may pass the host copy as a tuple; the layer passes the int64 tensor."""
    kda_recurrence = _kda_recurrence()
    cu_list = [0, 383, 785, 913, 1209]
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    inputs = _inputs(cu_list[-1], seed=580)
    upstream = torch.randn_like(inputs["v"])
    host = torch.tensor(cu_list, dtype=torch.int64)
    out_t, grads_t = _fwd_bwd(kda_recurrence, "deterministic", inputs, cu, host, upstream)
    out_s, grads_s = _fwd_bwd(kda_recurrence, "deterministic", inputs, cu, tuple(cu_list), upstream)
    out_f, grads_f = _fwd_bwd(kda_recurrence, "fla", inputs, cu, host, upstream)
    assert torch.equal(out_t, out_s)
    # the default alias runs the Cake forward: within the kernels' contract of fla's output, not its bits
    torch.testing.assert_close(out_t.float(), out_f.float(), atol=1e-2, rtol=1e-2)
    for name in _GRADS:
        assert torch.equal(grads_t[name], grads_s[name]), name
        torch.testing.assert_close(grads_t[name].float(), grads_f[name].float(), atol=1e-2, rtol=1e-2, msg=name)


def test_no_host_copy_fallback_matches_fla_bit_for_bit(monkeypatch):
    kda_recurrence = _kda_recurrence()
    monkeypatch.setattr(drop_in, "_fallback_warned", False)
    cu_list = [0, 100, 640]
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    inputs = _inputs(cu_list[-1], seed=581)
    upstream = torch.randn_like(inputs["v"])
    with pytest.warns(UserWarning, match="cu_seqlens_cpu"):
        out_d, grads_d = _fwd_bwd(kda_recurrence, "deterministic_fla_fwd", inputs, cu, None, upstream)
    out_f, grads_f = _fwd_bwd(kda_recurrence, "fla", inputs, cu, None, upstream)
    assert torch.equal(out_d, out_f)
    for name in _GRADS:
        assert torch.equal(grads_d[name], grads_f[name]), name


@pytest.mark.parametrize("name", sorted(PACKS))
def test_chunk_indices_equal_fla_prepare_chunk_indices(name):
    """The layout's ``chunk_indices`` are fla's ``prepare_chunk_indices`` of the same packing (the fla forward of the
    ``deterministic_fla_fwd`` mode consumes them in place of fla's own blocking upload)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    prepare_chunk_indices = pytest.importorskip("fla.ops.utils.index").prepare_chunk_indices
    cu_list = _cu(PACKS[name])
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    ref = prepare_chunk_indices(cu, CHUNK, cu_seqlens_cpu=torch.tensor(cu_list, dtype=torch.int64))
    det_backward._LAYOUT_CACHE.clear()
    layout = det_backward.chunk_layout(1, cu_list[-1], cu, cu.device, cu_seqlens_cpu=tuple(cu_list))
    assert layout.chunk_indices.dtype == ref.dtype and layout.chunk_indices.shape == ref.shape
    assert torch.equal(layout.chunk_indices, ref)


@pytest.mark.parametrize("backend", ["deterministic_fla_fwd", "deterministic_cake_fwd"])
def test_first_forward_of_a_new_pack_is_sync_free(backend):
    """U0-1 (c): the first forward of a never-seen packing uploads the layout (one pinned, non-blocking copy) and
    drains nothing in either mode: the fla forward takes its chunk indices from the layout instead of uploading
    its own (see the module docstring for what the sync debug mode flags)."""
    kda_recurrence = _kda_recurrence()
    inputs = _inputs(8192, seed=582)
    warm = _cu((2300, 3100, 2792))
    _fwd_only(kda_recurrence, backend, inputs, torch.tensor(warm, dtype=torch.int32, device="cuda"), torch.tensor(warm, dtype=torch.int64))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for n in range(1, 4):  # never-seen packings of the same token total
            cu_list = _cu((2300 + n, 3100 - n, 2792))
            cu = torch.tensor(cu_list, dtype=torch.int32).to("cuda", non_blocking=True)
            _fwd_only(kda_recurrence, backend, inputs, cu, torch.tensor(cu_list, dtype=torch.int64))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
