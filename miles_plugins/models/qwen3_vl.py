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

import torch

logger = logging.getLogger(__name__)

_PATCHED = "_miles_qwen3_vl_thd_mrope_patched"
_tls = threading.local()


def install_qwen3_vl_packed_mrope_patch() -> None:
    _patch_rotary_signature()
    _patch_model_forward_and_rope_index()


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

    Qwen3VLModel = getattr(model_mod, "Qwen3VLModel", None)
    if Qwen3VLModel is None or Qwen3VLModel.__dict__.get(_PATCHED, False):
        setattr(model_mod, _PATCHED, True)
        return

    orig_forward = Qwen3VLModel.forward

    def patched_forward(self, *args, **kwargs):
        packed = _build_packed_positions(self, args, kwargs, orig_get_rope_index)
        if packed is None:
            return orig_forward(self, *args, **kwargs)
        _tls.packed_positions = packed
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            _tls.packed_positions = None

    Qwen3VLModel.forward = patched_forward
    setattr(Qwen3VLModel, _PATCHED, True)
    setattr(model_mod, _PATCHED, True)


def _build_packed_positions(self, args, kwargs, orig_get_rope_index):
    # Per-segment MRoPE positions for a THD single-row packed batch; else None (run normally).
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
    cu = cu_t.detach().cpu().tolist()
    if cu[0] != 0 or cu[-1] != flat.numel():
        # cu_seqlens_q doesn't describe this local row. Under context parallelism the THD
        # row is laid out against cu_seqlens_q_padded (load-balanced CP chunks), which this
        # path does not yet reconstruct, so we fall back to the dense get_rope_index.
        # TODO(follow-up): per-segment positions for the CP + padded layout.
        logger.debug(
            "qwen3_vl packed mRoPE: cu_seqlens_q (%d) != local len (%d); using dense path", cu[-1], flat.numel()
        )
        return None

    image_grid_thw = kwargs.get("image_grid_thw")
    video_grid_thw = kwargs.get("video_grid_thw")
    merge = self.config.spatial_merge_size
    img_id, vid_id, vstart = self.image_token_id, self.video_token_id, self.vision_start_token_id

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
    return torch.cat(segments, dim=2).contiguous() if segments else None


def _slice(grid, offset, count):
    return None if grid is None or count == 0 else grid[offset : offset + count]
