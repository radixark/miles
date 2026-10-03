import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
except ImportError:
    _fla_chunk_gated_delta_rule = None


def get_chunk_gated_delta_rule(backend: str):
    if backend == "fla":
        if _fla_chunk_gated_delta_rule is None:
            raise ImportError("GDN backend 'fla' requires flash-linear-attention.")
        return _fla_chunk_gated_delta_rule

    if backend == "flashqla":
        try:
            from flash_qla import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError(
                "GDN backend 'flashqla' requires FlashQLA. Install it from https://github.com/QwenLM/FlashQLA."
            ) from exc
        return chunk_gated_delta_rule

    raise ValueError(f"Unsupported GDN backend: {backend}")


@functools.cache
def get_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as exc:
        raise ImportError("KDA requires flash-linear-attention >= 0.5 (fla.ops.kda).") from exc
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        os.environ.setdefault("FLA_TILELANG", "0")
    logger.info(f"KDA backward: FLA_TILELANG={os.environ.get('FLA_TILELANG', 'unset')}")
    return chunk_kda


@functools.cache
def get_short_conv_backend() -> str:
    backend = os.environ.get("FLA_CONV_BACKEND")
    if backend is None:
        try:
            from causal_conv1d.cpp_functions import causal_conv1d_bwd_function  # noqa: F401

            backend = "mix"
        except ImportError:
            backend = "triton"
    logger.info(f"Delta-rule short conv backend: {backend}")
    return backend
