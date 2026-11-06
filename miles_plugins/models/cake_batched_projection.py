"""The batched per-head projections of the MLA / DSA attention plugins, on ``torch.einsum`` or the Cake deterministic
GEMM of the miles-ext-kernels submodule (``--batched-projection-gemm-backend {torch,cake}``).

:func:`project_heads` is the q-absorb product ``torch.einsum("thd,hdm->thm", q, w_kc)`` (``weight_layout="in_out"``)
and the value product ``torch.einsum("thm,hdm->thd", o, w_vc)`` (``"out_in"``) of the GLM-5 / GLM-5.3 attention.
:func:`grouped_matmul` is DeepSeek-V4's grouped output projection ``torch.einsum("bsgd,grd->bsgr", x, w)``. Under
``cake`` both run ``miles_ext_kernels.cake.dense_gemm.batched_projection``: BF16 operands, FP32 accumulation, one
rounding, and a fixed reduction order, so the output and both gradients are bit-identical call to call (not bitwise
equal to cuBLAS). The views the einsums take (a slice of a wider head slot, a split of ``linear_kv_up_proj.weight``)
are read in place where the kernel's alignment allows, else copied.

:func:`validate_batched_projection_config` rejects at layer construction what the kernel does not run (CUDA graphs,
non-BF16 parameters, head dimensions that are not multiples of 8) and builds the kernel's programs there, so no
training step compiles. CUDA-only (NVIDIA SM100a / SM103a / SM107a, Megatron backend); the kernels are imported only
under ``cake``.
"""

from __future__ import annotations

import torch

BACKENDS = ("torch", "cake")
_EQUATIONS = {"in_out": "thd,hdm->thm", "out_in": "thm,hdm->thd"}


def projection_kernels():
    """The op module ``miles_ext_kernels.cake.dense_gemm``."""
    try:
        from miles_ext_kernels.cake import dense_gemm
    except ImportError as exc:
        raise ImportError(
            "--batched-projection-gemm-backend cake runs miles_ext_kernels.cake.dense_gemm from the miles-ext-kernels "
            "submodule: pass --use-miles-ext-kernels and check the submodule out with "
            "'git submodule update --init 3rdparty/miles-ext-kernels'."
        ) from exc
    return dense_gemm


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"batched projection backend must be one of {BACKENDS}, got {backend!r}")


def validate_batched_projection_config(
    config, *, num_heads: int, d_in: int, d_out: int, weight_layout: str, backend: str
) -> None:
    """Raise ``ValueError`` when ``backend`` is ``cake`` and the layer's Megatron ``config`` or its per-head projection
    (``num_heads`` heads of ``d_in -> d_out``, ``weight_layout`` as in :func:`project_heads`) is one the Cake kernel
    does not run; else, under ``cake``, build every program the projection may run on the current device, at any token
    count (``miles_ext_kernels.cake.prebuild``). Call it where the layer is built."""
    _check_backend(backend)
    if backend != "cake":
        return
    if config.cuda_graph_impl != "none":
        raise ValueError(
            "--batched-projection-gemm-backend cake keeps per-stream workspaces and plans on the host: it cannot "
            f"run under CUDA graphs (--cuda-graph-impl {config.cuda_graph_impl})"
        )
    if config.params_dtype != torch.bfloat16:
        raise ValueError(
            f"--batched-projection-gemm-backend cake takes BF16 weights; the parameters are {config.params_dtype}"
        )
    projection_kernels().check_projection_dims(d_in, d_out)
    from miles_ext_kernels import cake

    cake.prebuild("dense_gemm", projections=[(num_heads, d_in, d_out, weight_layout)])


def project_heads(x: torch.Tensor, weight: torch.Tensor, *, weight_layout: str, backend: str) -> torch.Tensor:
    """The per-head projection ``x [T, H, D_in] -> [T, H, D_out]``.

    ``weight_layout="in_out"``: ``weight [H, D_in, D_out]``, ``torch.einsum("thd,hdm->thm", x, weight)``.
    ``weight_layout="out_in"``: ``weight [H, D_out, D_in]``, ``torch.einsum("thm,hdm->thd", x, weight)``.
    """
    _check_backend(backend)
    if weight_layout not in _EQUATIONS:
        raise ValueError(f"weight_layout must be one of {tuple(_EQUATIONS)}, got {weight_layout!r}")
    if backend == "torch":
        return torch.einsum(_EQUATIONS[weight_layout], x, weight)
    return projection_kernels().batched_projection(x, weight, weight_layout=weight_layout)


def grouped_matmul(x: torch.Tensor, weight: torch.Tensor, *, backend: str) -> torch.Tensor:
    """Group ``g`` of ``x [b, s, g, d]`` times row block ``g`` of ``weight [g * r, d]``: ``[b, s, g, r]``, as
    ``torch.einsum("bsgd,grd->bsgr", x, weight.view(g, r, d))``."""
    _check_backend(backend)
    groups, depth = x.shape[2], weight.shape[-1]
    weight_grd = weight.view(groups, -1, depth)
    if backend == "torch":
        return torch.einsum("bsgd,grd->bsgr", x, weight_grd)
    out = project_heads(x.reshape(-1, groups, x.shape[3]), weight_grd, weight_layout="out_in", backend=backend)
    return out.view(*x.shape[:3], weight_grd.shape[1])
