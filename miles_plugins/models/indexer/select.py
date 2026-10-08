"""The one top-k entrypoint every sparse-attention indexer selects through.

Selection runs under `torch.no_grad` because the indexer is frozen on every
path, and always returns a fixed width so the sparse attention kernels see one
shape regardless of how many keys a query can reach.
"""

import torch

TOPK_BACKENDS = ("torch", "flashinfer")

# The slot a query does not attend to. Every sparse attention kernel masks it.
EMPTY_SLOT = -1

_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


def flashinfer_tie_break_value() -> int:
    """SGLang's flashinfer tie-break policy, so the trainer breaks ties as the rollout does."""
    from sglang.srt.environ import envs

    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]


def _torch_topk(logits: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    score, indices = torch.topk(logits, topk, dim=-1)
    indices = indices.to(torch.int32)
    return score, indices.masked_fill(score == -torch.inf, EMPTY_SLOT)


def _flashinfer_topk(logits: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    import flashinfer
    from sglang.srt.environ import envs

    orig_shape = logits.shape
    if logits.dim() > 2:
        logits = logits.reshape(-1, logits.shape[-1])

    score, indices = flashinfer.top_k(
        logits,
        topk,
        sorted=False,
        deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
        tie_break=flashinfer_tie_break_value(),
        dsa_graph_safe=True,
    )
    indices = indices.to(torch.int32)
    indices = indices.masked_fill(score == -torch.inf, EMPTY_SLOT)
    if len(orig_shape) > 2:
        indices = indices.reshape(*orig_shape[:-1], topk)
        score = score.reshape(*orig_shape[:-1], topk)
    return score, indices


_BACKENDS = {
    "torch": _torch_topk,
    "flashinfer": _flashinfer_topk,
}


@torch.no_grad()
def select_indexer_topk(
    logits: torch.Tensor,
    topk: int,
    *,
    backend: str = "torch",
    return_scores: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Select `topk` keys per query from indexer `logits`, as int32 indices.

    Keys the query must not see arrive as `-inf` and leave as `-1`, as do the
    slots left over when there are fewer than `topk` keys to choose from.
    `flashinfer` honors SGLang's determinism and tie-break environment
    variables; `return_scores` also returns the selected scores, for the
    kernels that read them back.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unsupported indexer topk backend {backend!r}, expected one of {TOPK_BACKENDS}.")
    if topk <= 0:
        raise ValueError(f"indexer topk must be positive, got {topk}.")

    width = min(topk, logits.shape[-1])
    scores, indices = _BACKENDS[backend](logits, width)

    if width < topk:
        tail = (*indices.shape[:-1], topk - width)
        indices = torch.cat([indices, indices.new_full(tail, EMPTY_SLOT)], dim=-1)
        scores = torch.cat([scores, scores.new_full(tail, -torch.inf)], dim=-1)

    return (scores, indices) if return_scores else indices


def get_indexer_topk_fn(backend: str):
    """A `(logits, topk) -> indices` callable bound to one backend."""
    if backend not in _BACKENDS:
        raise ValueError(f"Unsupported indexer topk backend {backend!r}, expected one of {TOPK_BACKENDS}.")

    def topk_fn(logits: torch.Tensor, topk: int) -> torch.Tensor:
        return select_indexer_topk(logits, topk, backend=backend)

    return topk_fn
