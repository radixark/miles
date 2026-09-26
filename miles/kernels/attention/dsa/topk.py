import torch


_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


def torch_dsa_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    score, indices = torch.topk(logits, topk, dim=-1)
    indices = indices.to(torch.int32)
    return indices.masked_fill(score == -torch.inf, -1)


def flashinfer_dsa_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
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
        tie_break=_flashinfer_tie_break_value(),
        dsa_graph_safe=True,
    )
    indices = indices.to(torch.int32)
    indices = indices.masked_fill(score == -torch.inf, -1)
    if len(orig_shape) > 2:
        indices = indices.reshape(*orig_shape[:-1], topk)
    return indices


def get_dsa_topk_fn(topk_backend: str):
    if topk_backend == "torch":
        return torch_dsa_topk
    if topk_backend == "flashinfer":
        return flashinfer_dsa_topk
    raise ValueError(f"Unsupported miles DSA topk backend: {topk_backend}")


def _flashinfer_tie_break_value() -> int:
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
