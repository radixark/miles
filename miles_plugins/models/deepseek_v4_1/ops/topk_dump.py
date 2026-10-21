"""Env-gated diagnostic dump of the DeepSeek-V4.1 indexer top-k selection (validation only).

Set ``MILES_DSV41_TOPK_DUMP_DIR`` to write, for sampled ``(rank, layer)`` calls of ``indexer_select``, the
configured backend's final indices next to the torch reference computed on the very same FP32 score rows
(``torch.topk`` -> ``-inf`` picks to ``-1`` -> ascending sort -> ``where(idx < lens, idx, -1)``, i.e. the
``torch`` path of the seam) and the top ``k + 1`` scores, so an end-to-end run can verify offline that the
cake top-k equals the torch path at every position that is not a tie at the cutoff -- without extra replays.
The hook runs for every backend, so a ``torch`` run produces ``idx == ref_idx`` files as a control.

Knobs (read once, cached): ``MILES_DSV41_TOPK_DUMP_LAYERS`` (comma-separated layer indices; default: all),
``MILES_DSV41_TOPK_DUMP_EVERY`` (dump every N-th call of a ``(rank, layer)``; default 50),
``MILES_DSV41_TOPK_DUMP_MAX`` (files per rank per layer; default 12).

File ``{DIR}/rank{r}_layer{l}_call{c}.pt`` holds a dict: ``schema`` (1), ``backend``, ``layer``, ``rank``,
``call``, ``time``, ``topk``, ``n_kv`` (the padded row width the backend saw), ``deep_select``,
``compress_lens`` (int64 ``[rows]``), ``idx`` / ``ref_idx`` (int32 ``[bsz, rows, topk]``, ``-1`` = empty) and
``topvals`` (FP32 ``[bsz * rows, min(topk + 1, n_kv)]``, descending).  Inert when the directory is unset;
I/O failures log one warning and never propagate.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

SCHEMA = 1


@dataclass
class TopkDumpConfig:
    directory: str
    layers: frozenset[int] | None
    every: int
    max_per_layer: int
    calls: dict[tuple[int, int], int] = field(default_factory=dict)
    written: dict[tuple[int, int], int] = field(default_factory=dict)
    warned: bool = False


@functools.lru_cache(maxsize=1)
def topk_dump_config() -> TopkDumpConfig | None:
    """The dump configuration from the environment, or ``None`` when ``MILES_DSV41_TOPK_DUMP_DIR`` is unset."""
    directory = os.environ.get("MILES_DSV41_TOPK_DUMP_DIR")
    if not directory:
        return None
    layers = frozenset(int(v) for v in os.environ.get("MILES_DSV41_TOPK_DUMP_LAYERS", "").split(",") if v.strip())
    return TopkDumpConfig(
        directory=directory,
        layers=layers or None,
        every=max(1, int(os.environ.get("MILES_DSV41_TOPK_DUMP_EVERY", "50"))),
        max_per_layer=int(os.environ.get("MILES_DSV41_TOPK_DUMP_MAX", "12")),
    )


def _rank() -> int:
    dist = torch.distributed
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def torch_reference(scores: torch.Tensor, lens: torch.Tensor, topk: int, shape) -> torch.Tensor:
    """The seam's ``torch`` path on ``scores`` ``[rows, n]``: ``torch.topk`` (``-inf`` picks -> ``-1``), padded to
    ``topk`` slots, sorted ascending (``-1`` first), reshaped to ``shape`` and masked with ``lens``."""
    n = scores.shape[-1]
    values, indices = torch.topk(scores, min(topk, n), dim=-1)
    ref = indices.masked_fill(values == -torch.inf, -1)
    if ref.shape[-1] < topk:
        ref = torch.nn.functional.pad(ref, (0, topk - ref.shape[-1]), value=-1)
    ref = ref.sort(dim=-1).values.reshape(shape)
    return torch.where(ref < lens.to(ref.device).unsqueeze(-1), ref, -1)


def maybe_dump_topk(
    backend: str,
    layer: int,
    scores: torch.Tensor,
    idx: torch.Tensor,
    lens: torch.Tensor,
    topk: int,
    *,
    deep_select: bool = False,
) -> str | None:
    """Write one diagnostic file for this ``(rank, layer)`` call when the sampling selects it; return its path.

    ``scores`` are the FP32 ``[rows, n]`` rows the backend saw, ``idx`` the seam's final indices
    ``[bsz, rows_per_batch, topk]`` (``-1`` = empty), ``lens`` the per-query valid lengths ``[rows_per_batch]``.
    Returns ``None`` when dumping is off, the call is not sampled, the per-layer budget is spent, or writing
    failed (logged once).
    """
    config = topk_dump_config()
    if config is None or (config.layers is not None and layer not in config.layers):
        return None
    rank = _rank()
    key = (rank, layer)
    call = config.calls.get(key, 0)
    config.calls[key] = call + 1
    if call % config.every or config.written.get(key, 0) >= config.max_per_layer:
        return None
    try:
        n = int(scores.shape[-1])
        payload = {
            "schema": SCHEMA,
            "backend": backend,
            "layer": int(layer),
            "rank": rank,
            "call": call,
            "time": time.time(),
            "topk": int(topk),
            "n_kv": n,
            "deep_select": bool(deep_select),
            "compress_lens": lens.detach().to(torch.int64).cpu(),
            "idx": idx.detach().to(torch.int32).cpu(),
            "ref_idx": torch_reference(scores, lens, topk, idx.shape).to(torch.int32).cpu(),
            "topvals": torch.topk(scores, min(topk + 1, n), dim=-1).values.float().cpu(),
        }
        os.makedirs(config.directory, exist_ok=True)
        path = os.path.join(config.directory, f"rank{rank}_layer{layer}_call{call}.pt")
        torch.save(payload, path)
        config.written[key] = config.written.get(key, 0) + 1
        return path
    except Exception as exc:  # diagnostics must never break the forward
        if not config.warned:
            config.warned = True
            logger.warning(
                "MILES_DSV41_TOPK_DUMP_DIR: could not write the top-k dump (%s); further errors are silent", exc
            )
        return None
