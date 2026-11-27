"""DeepSeek-V4.1 TOP across pipeline stages: what one stage hands the next besides the streams.

A V4.1 forward carries state across layers outside the hidden streams: the previous hyper-connection's ``pre``, each
KV source's reference latent (both differentiated), and the trainer's SGLang attention runtime (the KV sources'
compressed caches, the index layers' K tables and selections, the candidate source's mask), which later layers read as
one serving prefill would. A non-last stage appends that state to the streams it sends, as extra rows of raw bytes
under a header row (Megatron exchanges each activation's shape, so the rows may vary per micro-batch), and the next
stage restores it before its first layer reads it: ``pre`` and the latents as differentiable tensors whose gradients
travel back as bytes in the same rows, the runtime tensors byte for byte once the stage has prepared the micro-batch's
runtime. Every stage prepares the same micro-batches in the same order, so the runtimes match in shape.
"""

from __future__ import annotations

import math

import torch

_MAGIC = 0x7641505053544154  # "vAPPSTAT"
_ALIGN = 16
# Header: magic, rows of streams, number of segments, then per segment its fields.
_FIELDS = ("kind", "key", "dtype", "ndim", "d0", "d1", "d2", "d3", "offset", "nbytes")
_PRE, _LATENT, _RUNTIME = 0, 1, 2
_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


class Inbound:
    """What the previous stage sent for the coming forward, until the block state and the runtime take it."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.pre = None
        self.latents = {}
        # (kind, key) of the differentiable segments _Unpack returns, in order; the runtime's (key, tensor).
        self.keys = []
        self.runtime = None


INBOUND = Inbound()


def runtime_tensors(rt) -> list[tuple[int, torch.Tensor]]:
    """(key, tensor) of the runtime state a later layer reads from an earlier one, in a fixed order: the ratio-1/2 KV
    sources' compressed caches, the index K tables, the sparse page and raw selections, and the candidate mask."""
    pool, core = rt.pool, rt.backend.forward_metadata.core_metadata
    out = []
    for ratio in (1, 2):
        kv_pool = pool.kv_pools.get(ratio)
        if kv_pool is not None:
            out += [(ratio * 1000 + j, buf) for j, buf in enumerate(kv_pool.kv_buffer)]
        index_pool = pool.index_pools.get(ratio)
        if index_pool is not None:
            out += [(ratio * 1000 + 100 + j, buf) for j, buf in enumerate(index_pool.contiguous_page_row_buffers())]
        for slot, get in ((200, core.sparse_page_indices), (300, core.sparse_raw_indices)):
            selection = get(ratio)
            if selection is not None:
                out.append((ratio * 1000 + slot, selection))
    if rt.backend.top_candidates is not None:
        out.append((9000, rt.backend.top_candidates))
    return out


def _bytes(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    return (t.view(torch.uint8) if t.dtype != torch.bool else t.to(torch.uint8)).reshape(-1)


def _from_bytes(raw: torch.Tensor, dtype: torch.dtype, shape) -> torch.Tensor:
    if dtype == torch.bool:
        return raw.clone().to(torch.bool).view(shape)
    return raw.clone().view(dtype).view(shape)


def _layout(segments):
    """Byte offsets of the segments, aligned, and the header rows' int64 values."""
    entries, offset = [], 0
    for kind, key, t in segments:
        if t.ndim > 4:
            raise NotImplementedError("[top] V4.1 pipeline state tensors have at most 4 dims")
        nbytes = t.numel() * (1 if t.dtype == torch.bool else t.element_size())
        dims = list(t.shape) + [0] * (4 - t.ndim)
        entries.append([kind, key, _DTYPES.index(t.dtype), t.ndim, *dims, offset, nbytes])
        offset += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    return entries, offset


class _Pack(torch.autograd.Function):
    """streams [s, b, w] + segments -> [s + k + 1, b, w]: the payload rows, then the header row."""

    @staticmethod
    def forward(ctx, streams, kinds_keys, runtime, *diff):
        segments = [(kind, key, t) for (kind, key), t in zip(kinds_keys, diff, strict=True)]
        segments += [(_RUNTIME, key, t) for key, t in runtime]
        entries, payload = _layout(segments)
        s, b, w = streams.shape
        row = b * w * streams.element_size()
        header = torch.tensor([_MAGIC, s, len(entries), *[v for e in entries for v in e]], dtype=torch.int64)
        if header.numel() * 8 > row:
            raise NotImplementedError("[top] V4.1 pipeline state header does not fit one row")
        rows = math.ceil(payload / row)
        out = streams.new_empty((s + rows + 1, b, w))
        out[:s] = streams
        body = out[s:].view(-1).view(torch.uint8)
        body.zero_()
        for (_, _, t), e in zip(segments, entries, strict=True):
            body[e[-2] : e[-2] + e[-1]] = _bytes(t)
        body[rows * row : rows * row + header.numel() * 8] = header.to(out.device).view(torch.uint8)
        ctx.layout = (s, rows, row, entries[: len(diff)])
        return out

    @staticmethod
    def backward(ctx, grad):
        s, rows, row, entries = ctx.layout
        body = grad[s:].contiguous().view(-1).view(torch.uint8)
        grads = []
        for e in entries:
            shape = e[4 : 4 + e[3]]
            grads.append(_from_bytes(body[e[-2] : e[-2] + e[-1]], _DTYPES[e[2]], shape))
        return (grad[:s], None, None, *grads)


class _Unpack(torch.autograd.Function):
    """The inverse of _Pack: (streams, *differentiable segments); the runtime bytes go to INBOUND."""

    @staticmethod
    def forward(ctx, packed):
        b, w = packed.shape[1:]
        row = b * w * packed.element_size()
        last = packed[-1].contiguous().view(-1).view(torch.uint8)
        head = last[:24].view(torch.int64).tolist()
        if head[0] != _MAGIC:
            raise RuntimeError("[top] the previous V4.1 stage sent no pipeline state header")
        s, count = head[1], head[2]
        values = last[: (3 + count * len(_FIELDS)) * 8].view(torch.int64).tolist()[3:]
        entries = [values[i * len(_FIELDS) : (i + 1) * len(_FIELDS)] for i in range(count)]
        body = packed[s:-1].contiguous().view(-1).view(torch.uint8)
        diff, runtime = [], []
        for e in entries:
            t = _from_bytes(body[e[-2] : e[-2] + e[-1]], _DTYPES[e[2]], e[4 : 4 + e[3]])
            (runtime if e[0] == _RUNTIME else diff).append((e[0], e[1], t))
        ctx.layout = (packed.shape, s, row, [e for e in entries if e[0] != _RUNTIME])
        INBOUND.keys = [(kind, key) for kind, key, _ in diff]
        INBOUND.runtime = [(key, t) for _, key, t in runtime]
        return (packed[:s].clone(), *[t for _, _, t in diff])

    @staticmethod
    def backward(ctx, grad_streams, *grads):
        shape, s, row, entries = ctx.layout
        out = grad_streams.new_zeros(shape)
        out[:s] = grad_streams
        body = out[s:-1].view(-1).view(torch.uint8)
        for e, g in zip(entries, grads, strict=True):
            if g is not None:
                body[e[-2] : e[-2] + e[-1]] = _bytes(g.to(_DTYPES[e[2]]))
        return out


def pack(streams: torch.Tensor) -> torch.Tensor:
    """The stage output Megatron sends: the streams, then ``pre``, the reference latents of this forward and the
    runtime state."""
    from miles_plugins.top.dsv41_attention import _SOURCE_LATENTS
    from miles_plugins.top.dsv41_runtime import _RUNTIME as rt
    from miles_plugins.top.dsv41_runtime import BLOCK_STATE

    if BLOCK_STATE.pre is None or rt is None:
        raise RuntimeError("[top] a V4.1 stage ended without a hyper-connection pre or an attention runtime")
    kinds_keys = [(_PRE, 0)]
    diff = [BLOCK_STATE.pre]
    for ratio, latent in sorted(_SOURCE_LATENTS.items()):
        kinds_keys.append((_LATENT, ratio))
        diff.append(latent)
    return _Pack.apply(streams, kinds_keys, runtime_tensors(rt), *diff)


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """The streams the previous stage sent; ``pre``, the latents and the runtime bytes wait in INBOUND."""
    INBOUND.clear()
    streams, *diff = _Unpack.apply(packed)
    for (kind, key), t in zip(INBOUND.keys, diff, strict=True):
        if kind == _PRE:
            INBOUND.pre = t
        else:
            INBOUND.latents[key] = t
    if INBOUND.pre is None:
        raise RuntimeError("[top] the previous V4.1 stage sent no hyper-connection pre")
    return streams


def restore_block_state():
    """At the forward's start: ``pre`` and the latents the previous stage sent (none on the first stage)."""
    from miles_plugins.top.dsv41_attention import _SOURCE_LATENTS
    from miles_plugins.top.dsv41_runtime import BLOCK_STATE

    _SOURCE_LATENTS.clear()
    _SOURCE_LATENTS.update(INBOUND.latents)
    BLOCK_STATE.pre = INBOUND.pre
    INBOUND.pre, INBOUND.latents = None, {}


def restore_runtime(rt):
    """After the stage's first prepare: the earlier stages' runtime state, byte for byte."""
    if INBOUND.runtime is None:
        return
    received, INBOUND.runtime = INBOUND.runtime, None
    local = dict(runtime_tensors(rt))
    for key, t in received:
        if key == 9000:
            rt.backend.top_candidates = t
            continue
        if key not in local or local[key].shape != t.shape or local[key].dtype != t.dtype:
            raise RuntimeError(f"[top] V4.1 stage runtimes disagree on state {key}: the stages prepared differently")
        local[key].copy_(t)
    if 9000 not in dict(received):
        rt.backend.top_candidates = None


# What may cross a stage boundary: the hyper-connection pre alone; a ratio-2 KV source's latent, cache, index K and
# selection with its readers in the next stage; the same for ratio 1 together with the candidate source's mask.
QUALIFIED_CROSSINGS = (frozenset({"pre"}), frozenset({"pre", "ratio2"}), frozenset({"pre", "ratio1", "candidates"}))


def crossing(cfg, start: int) -> frozenset[str]:
    """The state layers before ``start`` produce that layers from ``start`` on read."""
    ratios = cfg.compress_ratios[: cfg.num_hidden_layers]
    kinds = {"pre"}
    for layer in range(start, cfg.num_hidden_layers):
        ratio = ratios[layer]
        if ratio not in (1, 2):
            continue
        sources = [i for i in cfg.kv_source_layer_ids if ratios[i] == ratio and i <= layer]
        indexes = [i for i in cfg.index_source_layer_ids if ratios[i] == ratio and i <= layer]
        if (sources and max(sources) < start) or (indexes and max(indexes) < start):
            kinds.add(f"ratio{ratio}")
        candidate = getattr(
            cfg, "candidate_source_layer_id", None
        )  # config-access-exempt: TransformerConfig versions and model families differ in optional candidate_source_layer_id
        if candidate is not None and candidate < start <= layer and layer in cfg.index_source_layer_ids:
            kinds.add("candidates")
    return frozenset(kinds)


def install(model):
    """Wire one stage of a V4.1 GPTModel: unpack what the previous stage sent, pack what the next one needs."""
    from sglang.srt.true_on_policy.registry import NOT_QUALIFIED

    from miles_plugins.top.dsv4_runtime import dsv4_hf_config

    decoder = model.decoder
    if not model.pre_process:
        start = decoder.layers[0].layer_number - 1
        kinds = crossing(dsv4_hf_config(), start)
        if kinds not in QUALIFIED_CROSSINGS:
            raise NotImplementedError(
                f"[top] DeepSeek-V4.1 stage boundary before layer {start} carries {sorted(kinds)}: {NOT_QUALIFIED}"
            )
        set_input = decoder.set_input_tensor

        def set_input_tensor(tensor):
            set_input(unpack(tensor) if tensor is not None else None)

        decoder.set_input_tensor = set_input_tensor
    if not model.post_process:

        def outbound(module, args, output):
            if not isinstance(output, torch.Tensor):
                raise NotImplementedError("[top] a V4.1 non-last stage must return its streams as one tensor")
            return pack(output)

        decoder.register_forward_hook(outbound)
