"""miles-side fix for Qwen3-VL THD packed mRoPE positions (no Megatron-Bridge edit).

Bridge's Qwen3VLModel.forward resets position_ids=None and recomputes via the
module-level get_rope_index over the whole [1, total] packed row, so MRoPE positions
don't restart per packed segment (wrong for multimodal). We hijack that call: stash
correct per-segment positions and have a patched get_rope_index return them.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

_PATCHED = "_miles_qwen3_vl_thd_mrope_patched"
_tls = threading.local()


def install_qwen3_vl_packed_mrope_patch() -> None:
    _patch_rotary_signature()
    _patch_model_forward_and_rope_index()
    _patch_allgather_vision_embeddings_kwarg()


def _patch_allgather_vision_embeddings_kwarg() -> None:
    """megatron-bridge 0.5.0 calls AllGatherVisionEmbeddings.apply(..., cp_group=...) in the
    Qwen3-VL vision_dp_when_cp path, but torch.autograd.Function.apply rejects keyword args
    (TypeError: apply() takes no keyword arguments). Replace the symbol with a shim whose
    .apply accepts cp_group as a kwarg and forwards it positionally.
    """
    try:
        model_mod = importlib.import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")
    except ImportError:
        return
    orig = getattr(model_mod, "AllGatherVisionEmbeddings", None)
    if orig is None or getattr(orig, "_miles_kwarg_shim", False):
        return

    class _AllGatherVisionEmbeddingsKwargShim:
        _miles_kwarg_shim = True

        @staticmethod
        def apply(input, seqlens_on_cp_ranks, cp_group=None):
            return orig.apply(input, seqlens_on_cp_ranks, cp_group)

    model_mod.AllGatherVisionEmbeddings = _AllGatherVisionEmbeddingsKwargShim


def _patch_rotary_signature() -> None:
    # Let the rotary embedding forward tolerate the packed_seq_params kwarg.
    try:
        text_model = importlib.import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model")
    except ImportError:
        return
    for name in ("Qwen3VLTextRotaryEmbedding", "Qwen3VLMoETextRotaryEmbedding"):
        cls = getattr(text_model, name, None)
        if cls is None or cls.__dict__.get(_PATCHED, False):
            continue
        _orig = cls.forward

        def _make(orig):
            def _fwd(self, *args, packed_seq_params=None, **kwargs):
                return orig(self, *args, **kwargs)

            return _fwd

        cls.forward = _make(_orig)
        setattr(cls, _PATCHED, True)


def _patch_model_forward_and_rope_index() -> None:
    try:
        model_mod = importlib.import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")
    except ImportError:
        return
    if getattr(model_mod, _PATCHED, False):
        return

    orig_get_rope_index = model_mod.get_rope_index

    def patched_get_rope_index(*args, **kwargs):
        pending = getattr(_tls, "packed_positions", None)
        if pending is not None:
            return pending, None
        return orig_get_rope_index(*args, **kwargs)

    model_mod.get_rope_index = patched_get_rope_index

    # Under CP, miles pre-shards the THD row (slice_with_cp), but the bridge forward re-shards
    # internally via preprocess_packed_seqs (it expects the FULL input). When miles already
    # sharded, make that internal call an identity that returns miles' packed_seq_params (so CP
    # attention still sees the full cu_seqlens) instead of re-splitting the already-local data.
    _patch_preprocess_packed_seqs_identity(model_mod)

    Qwen3VLModel = getattr(model_mod, "Qwen3VLModel", None)
    # The bridge selects CP-local vision embeds natively; warn when running an old one.
    if Qwen3VLModel is not None and not hasattr(Qwen3VLModel, "_cp_local_vision_embed_indices"):
        logger.warning(
            "megatron-bridge Qwen3VLModel lacks native CP-local vision-embed selection; "
            "CP runs with vision tokens will mis-place vision embeddings. "
            "Apply the matching Megatron-Bridge patch (radixark/Megatron-Bridge PR #9)."
        )

    if Qwen3VLModel is None or Qwen3VLModel.__dict__.get(_PATCHED, False):
        setattr(model_mod, _PATCHED, True)
        return

    orig_forward = Qwen3VLModel.forward

    def patched_forward(self, *args, **kwargs):
        parsed = _parse_packed_thd(args, kwargs)
        packed = _build_packed_positions(self, parsed, kwargs, orig_get_rope_index)
        ctx = _prepare_cp_local_context(parsed)
        if packed is not None:
            _tls.packed_positions = packed
        if ctx is not None:
            _tls.cp_local = ctx
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            _tls.packed_positions = None
            _tls.cp_local = None

    Qwen3VLModel.forward = patched_forward
    setattr(Qwen3VLModel, _PATCHED, True)
    setattr(model_mod, _PATCHED, True)


def _patch_preprocess_packed_seqs_identity(model_mod) -> None:
    orig = getattr(model_mod, "preprocess_packed_seqs", None)
    if orig is None or getattr(orig, "_miles_identity_wrapped", False):
        return

    def wrapped(input_ids, attention_mask, *args, **kwargs):
        ctx = getattr(_tls, "cp_local", None)
        if ctx is not None:
            # Already CP-local: skip re-sharding and hand back miles' full-cu packed_seq_params.
            return input_ids, ctx["psp"]
        return orig(input_ids, attention_mask, *args, **kwargs)

    wrapped._miles_identity_wrapped = True
    model_mod.preprocess_packed_seqs = wrapped


class _PackedTHD(NamedTuple):
    """Parsed view of a THD single-row packed batch (see ``_parse_packed_thd``)."""

    psp: object
    flat: torch.Tensor
    cu: list  # cu_seqlens_q as a host-side list (one GPU->host copy, shared by both paths)
    cu_t: torch.Tensor
    local_len: int
    cp_size: int
    cp_rank: int


def _parse_packed_thd(args, kwargs):
    """Extract the preamble shared by the mRoPE-position and CP-context paths, doing the one
    GPU->host copy of cu_seqlens_q a single time. Returns None when this is not a THD packed
    ``[1, T]`` batch, in which case both callers run the dense / unchanged path."""
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    psp = kwargs.get("packed_seq_params")
    if psp is None or getattr(psp, "qkv_format", None) != "thd":
        return None
    if input_ids is None or input_ids.dim() != 2 or input_ids.shape[0] != 1:
        return None
    cu_t = getattr(psp, "cu_seqlens_q", None)
    if cu_t is None or cu_t.numel() < 2:
        return None
    flat = input_ids.reshape(-1)
    cp_size, cp_rank = _cp_size_rank()
    return _PackedTHD(psp, flat, cu_t.detach().cpu().tolist(), cu_t, flat.numel(), cp_size, cp_rank)


def _prepare_cp_local_context(parsed):
    """When CP has already pre-sharded this THD row, capture miles' packed_seq_params so the
    preprocess_packed_seqs identity wrapper can hand them back unchanged (the bridge does
    CP-local vision-embed selection natively). Returns None for the non-CP / full-input cases
    (bridge runs unchanged)."""
    if parsed is None or parsed.cp_size <= 1 or parsed.cu[0] != 0:
        return None
    if parsed.cu[-1] != parsed.cp_size * parsed.local_len:
        return None
    return {"psp": parsed.psp}


def _cp_size_rank():
    """Context-parallel (size, rank); (1, 0) when CP is unavailable."""
    try:
        from megatron.core import parallel_state as _ps

        return _ps.get_context_parallel_world_size(), _ps.get_context_parallel_rank()
    except Exception:
        return 1, 0


def _natural_to_zigzag_slice(t, cp_size, cp_rank, dim):
    """Slice a full-length tensor into this rank's zigzag (load-balanced ring-attn) CP chunks.

    Mirrors miles.backends.training_utils.cp_utils.slice_with_cp / natural_to_zigzag_slice:
    rank r owns chunks [r, 2*cp_size-1-r] of the 2*cp_size equal partitions along ``dim``.
    """
    total = t.shape[dim]
    num_chunks = 2 * cp_size
    chunk = total // num_chunks
    idxs = [cp_rank, 2 * cp_size - 1 - cp_rank]
    return torch.cat([t.narrow(dim, i * chunk, chunk) for i in idxs], dim=dim)


def _cp_allgather_unzigzag(flat, cu, cp_size):
    """Reconstruct the full THD packed row from this rank's zigzag chunks.

    Under CP, ``flat`` holds only chunks [cp_rank, 2*cp-1-cp_rank] of every segment, and
    ``cu`` (== psp.cu_seqlens_q) gives the FULL padded per-segment boundaries. All-gather the
    per-rank rows over the CP group and de-interleave each segment back to natural order.
    Returns None (caller falls back to dense) if a segment is not divisible by 2*cp.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state as _ps

    group = _ps.get_context_parallel_group()
    gathered = [torch.empty_like(flat) for _ in range(cp_size)]
    dist.all_gather(gathered, flat.contiguous(), group=group)
    return _reassemble_full_row(gathered, cu, cp_size)


def _reassemble_full_row(gathered, cu, cp_size):
    """De-interleave per-rank zigzag rows back into the full natural-order packed row.

    ``gathered[r]`` is rank r's local row; ``cu`` (full padded per-segment boundaries, i.e.
    miles' cu_seqlens * cp) locates each segment. For segment ``i`` of full length ``L`` (a
    multiple of 2*cp), rank r contributed chunk r and chunk 2*cp-1-r, each of size L/(2*cp),
    at local offset cu[i]//cp. Pure (no collectives) so it is unit-testable. Returns None if a
    segment is not divisible by 2*cp (caller falls back to the dense path).
    """
    full = torch.zeros(cu[-1], dtype=gathered[0].dtype, device=gathered[0].device)
    for i in range(len(cu) - 1):
        seg_full = cu[i + 1] - cu[i]
        if seg_full <= 0:
            continue
        if seg_full % (2 * cp_size) != 0:
            return None
        c = seg_full // (2 * cp_size)
        local_off = cu[i] // cp_size  # this segment's offset within a per-rank (local) row
        for r in range(cp_size):
            mir = 2 * cp_size - 1 - r
            full[cu[i] + r * c : cu[i] + (r + 1) * c] = gathered[r][local_off : local_off + c]
            full[cu[i] + mir * c : cu[i] + (mir + 1) * c] = gathered[r][local_off + c : local_off + 2 * c]
    return full


def _segment_positions(model, flat, cu, cu_t, kwargs, orig_get_rope_index):
    """Per-segment MRoPE positions for a full (unsharded) packed row `flat` with boundaries `cu`.

    Returns a list of [3, 1, seg_len] tensors (one per non-empty segment), text segments get
    a linear 0..L range, media segments call get_rope_index with the matching grid slice.
    """
    image_grid_thw = kwargs.get("image_grid_thw")
    video_grid_thw = kwargs.get("video_grid_thw")
    merge = model.config.spatial_merge_size
    img_id, vid_id, vstart = model.image_token_id, model.video_token_id, model.vision_start_token_id

    # Vectorized media count per segment (one GPU->host copy total, no per-segment .item()).
    num_segments = len(cu) - 1
    img_counts = [0] * num_segments
    vid_counts = [0] * num_segments
    starts = torch.nonzero(flat == vstart, as_tuple=False).flatten()
    starts = starts[starts + 1 < flat.numel()]
    if starts.numel() > 0:
        toks = flat[starts + 1]
        seg_idx = torch.bucketize(starts, cu_t, right=True) - 1  # [start,end) -> segment index
        img_counts = torch.bincount(seg_idx[toks == img_id], minlength=num_segments).cpu().tolist()
        vid_counts = torch.bincount(seg_idx[toks == vid_id], minlength=num_segments).cpu().tolist()

    img_off = vid_off = 0
    segments = []
    for i, (start, end) in enumerate(zip(cu[:-1], cu[1:], strict=False)):
        if end <= start:
            continue
        seg = flat[start:end]
        ic, vc = img_counts[i], vid_counts[i]
        if ic == 0 and vc == 0:
            pos = torch.arange(seg.numel(), dtype=seg.dtype, device=seg.device).view(1, 1, -1).expand(3, 1, -1)
        else:
            pos, _ = orig_get_rope_index(
                merge,
                img_id,
                vid_id,
                vstart,
                seg.unsqueeze(0),
                image_grid_thw=_slice(image_grid_thw, img_off, ic),
                video_grid_thw=_slice(video_grid_thw, vid_off, vc),
                attention_mask=None,
            )
            pos = pos[:, :, : seg.numel()]
        img_off += ic
        vid_off += vc
        segments.append(pos)
    return segments


def _build_packed_positions(model, parsed, kwargs, orig_get_rope_index):
    """Per-segment MRoPE positions for a THD packed batch; None to run the dense path."""
    if parsed is None or parsed.cu[0] != 0:
        return None
    flat, cu, cu_t = parsed.flat, parsed.cu, parsed.cu_t
    local_len, cp_size, cp_rank = parsed.local_len, parsed.cp_size, parsed.cp_rank

    # Non-CP (or single chunk): cu_seqlens_q already describes this row exactly.
    if cu[-1] == local_len:
        segments = _segment_positions(model, flat, cu, cu_t, kwargs, orig_get_rope_index)
        return torch.cat(segments, dim=2).contiguous() if segments else None

    # CP + THD packing: cu_seqlens_q gives the FULL padded per-segment boundaries (miles
    # builds cu_seqlens * cp_size), while this row holds only this rank's zigzag chunks
    # (full_len / cp). Reconstruct the full row across the CP group, build full per-segment
    # MRoPE positions, then re-slice each segment into this rank's zigzag layout so the
    # positions line up with the tokens that slice_with_cp produced.
    if cp_size > 1 and cu[-1] == cp_size * local_len:
        full_flat = _cp_allgather_unzigzag(flat, cu, cp_size)
        if full_flat is None:
            logger.debug("qwen3_vl packed mRoPE: CP segment not divisible by 2*cp; dense path")
            return None
        segments = _segment_positions(model, full_flat, cu, cu_t, kwargs, orig_get_rope_index)
        if not segments:
            return None
        local_segments = [_natural_to_zigzag_slice(p, cp_size, cp_rank, dim=2) for p in segments]
        return torch.cat(local_segments, dim=2).contiguous()

    # Unrecognized layout -> let the dense get_rope_index run.
    logger.debug(
        "qwen3_vl packed mRoPE: cu_seqlens_q (%d) vs local len (%d), cp=%d; dense path",
        cu[-1],
        local_len,
        cp_size,
    )
    return None


def _slice(grid, offset, count):
    return None if grid is None or count == 0 else grid[offset : offset + count]
