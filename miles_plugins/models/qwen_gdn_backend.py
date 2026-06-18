try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
except ImportError:
    _fla_chunk_gated_delta_rule = None


def get_chunk_gated_delta_rule(backend: str):
    if backend == "fla":
        if _fla_chunk_gated_delta_rule is None:
            raise ImportError("Qwen GDN backend 'fla' requires flash-linear-attention.")
        return _fla_chunk_gated_delta_rule

    if backend == "flashqla":
        try:
            from flash_qla import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError(
                "Qwen GDN backend 'flashqla' requires FlashQLA. Install it from https://github.com/QwenLM/FlashQLA."
            ) from exc
        return chunk_gated_delta_rule

    raise ValueError(f"Unsupported Qwen GDN backend: {backend}")
