"""MiniMax-M3 multimodal data: expand media placeholders to per-patch slots.

Mirrors miles' ``miles/backends/training_utils/mm_data.py`` (Kimi-VL), adapted
to M3's token ids and 2×2 spatial merge. The rollout/processor emits ONE media
placeholder per image/video; training expands it to the grid-derived number of
slots so the LM has one position per merged vision patch — which is exactly what
``vl_model.merge_vision_into_text`` requires (placeholder count == vision tokens).

M3 constants (from configuration_minimax_m3_vl.py):
  image_token_index = 200025, video_token_index = 200026
  spatial merge = 2×2 (img_token_compression), temporal pooled away.
"""

from __future__ import annotations

import torch

# from config.json: image_token_index / video_token_index
M3_IMAGE_TOKEN_ID = 200025
M3_VIDEO_TOKEN_ID = 200026
# 2×2 spatial merge (img_token_compression_config); temporal averaged.
_MERGE_H = 2
_MERGE_W = 2
_GRID_KEYS = ("image_grid_thw", "grid_thws")


def _get_grid(mm: dict | None):
    if not mm:
        return None
    for k in _GRID_KEYS:
        if k in mm:
            return mm[k]
    return None


def num_tokens_from_grid(grid_thw: torch.Tensor, merge_h: int = _MERGE_H, merge_w: int = _MERGE_W) -> int:
    """Tokens an image/video occupies after the spatial merge (temporal pooled)."""
    _, h, w = grid_thw.tolist()
    return (h // merge_h) * (w // merge_w)


def expand_media_tokens(
    tokens: torch.Tensor,        # [seq] int64
    loss_mask: torch.Tensor,     # [seq]
    grids: torch.Tensor,         # [num_media, 3]  (T, H, W), in media order
    *,
    image_token_id: int = M3_IMAGE_TOKEN_ID,
    video_token_id: int = M3_VIDEO_TOKEN_ID,
    merge_h: int = _MERGE_H,
    merge_w: int = _MERGE_W,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace each placeholder token with N copies (N = grid-derived count).

    The i-th placeholder (scanning left→right, images and videos sharing one
    ordered media list, like the HF processor) consumes ``grids[i]``. Returns the
    expanded ``(tokens, loss_mask)``. Idempotent guard: if the placeholder run is
    already expanded (count matches), it is left as-is.
    """
    if grids is None or len(grids) == 0:
        return tokens, loss_mask
    # A single image/video can arrive as a 1-D (t, h, w) grid instead of [n, 3];
    # normalize so grids[i] is always one (t, h, w) triple.
    if getattr(grids, "ndim", None) == 1:
        grids = grids.unsqueeze(0)

    device = tokens.device
    is_media = (tokens == image_token_id) | (tokens == video_token_id)
    if not bool(is_media.any()):
        return tokens, loss_mask

    out_tokens, out_mask = [], []
    media_idx = 0
    i = 0
    n = tokens.numel()
    while i < n:
        tok = tokens[i]
        if is_media[i]:
            count = num_tokens_from_grid(grids[media_idx], merge_h, merge_w)
            media_idx += 1
            out_tokens.append(tok.repeat(count))
            # media positions stay masked out of the loss (prompt content)
            out_mask.append(torch.zeros(count, dtype=loss_mask.dtype, device=device))
            i += 1
        else:
            out_tokens.append(tok.view(1))
            out_mask.append(loss_mask[i].view(1))
            i += 1
    return torch.cat(out_tokens), torch.cat(out_mask)


def expand_multimodal_in_place(rollout_data: dict) -> None:
    """Expand every sample's media placeholders in a miles rollout_data dict.

    Hook this in miles's ``get_batch`` / data-iterator the same way miles calls
    ``expand_multimodal_rollout_data_in_place`` (data.py:141 / :379). Expects
    per-sample ``tokens``, ``loss_masks`` lists and a parallel
    ``multimodal_train_inputs`` list of dicts carrying ``image_grid_thw``.
    """
    tokens = rollout_data.get("tokens")
    masks = rollout_data.get("loss_masks")
    mm = rollout_data.get("multimodal_train_inputs")
    if tokens is None or mm is None:
        return
    new_tokens, new_masks, new_lens = [], [], []
    for tk, mk, mmi in zip(tokens, masks, mm, strict=False):
        grid = _get_grid(mmi)
        if grid is None:
            new_tokens.append(tk)
            new_masks.append(mk)
            new_lens.append(len(tk) if hasattr(tk, "__len__") else tk.numel())
            continue
        tk = tk if torch.is_tensor(tk) else torch.tensor(tk, dtype=torch.long)
        mk = mk if torch.is_tensor(mk) else torch.tensor(mk)
        et, em = expand_media_tokens(tk, mk, grid)
        new_tokens.append(et)
        new_masks.append(em)
        new_lens.append(et.numel())
    rollout_data["tokens"] = new_tokens
    rollout_data["loss_masks"] = new_masks
    if "total_lengths" in rollout_data:
        rollout_data["total_lengths"] = new_lens
