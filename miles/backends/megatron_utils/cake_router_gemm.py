"""Megatron's MoE router gate GEMM on the Cake FP32 router kernels of the miles-ext-kernels submodule
(``--moe-router-gemm-backend cake``).

:func:`install_cake_router_gemm` replaces ``Router.gating`` of every MoE router of a built Megatron model with
``miles_ext_kernels.dense_gemm.router_gating``. It computes what Megatron's ``RouterGatingLinearFunction`` computes
under ``--moe-router-dtype fp32`` without a bias: FP32 logits from the BF16 input and weight, and the input and weight
gradients as FP32 GEMMs rounded to the input's and the weight's dtype. Its reductions run in a fixed order, so the
logits and both gradients are bit-identical call to call; they are not bitwise equal to Transformer Engine's GEMM.
Megatron's ``--moe-router-use-torch-mm`` (which miles sets for every run) only selects the GEMM inside
``RouterGatingLinearFunction`` (``torch.mm`` of the operands cast to FP32 instead of Transformer Engine's GEMM of the
BF16 operands); both compute the same FP32 products and sums, and the replaced ``Router.gating`` runs neither.

The router must use Megatron's ``Router.gating`` (``TopKRouter`` and subclasses that do not override it), FP32 router
dtype, no router bias (``--disable-bias-linear``), BF16 parameters, and no CUDA graphs; :func:`validate_router_config`
rejects anything else where the model is built. Expert counts and hidden sizes must be multiples of 8. CUDA-only
(NVIDIA SM100a / SM103a, Megatron backend): miles imports Megatron and the kernel package here only when it installs
the backend.
"""

from __future__ import annotations

import logging
import types

import torch

logger = logging.getLogger(__name__)


def router_kernels():
    """The ``miles_ext_kernels.dense_gemm`` kernel package."""
    try:
        from miles_ext_kernels import dense_gemm
    except ImportError as exc:
        raise ImportError(
            "--moe-router-gemm-backend cake runs miles_ext_kernels.dense_gemm from the miles-ext-kernels submodule: "
            "pass --use-miles-ext-kernels and check the submodule out with "
            "'git submodule update --init 3rdparty/miles-ext-kernels'."
        ) from exc
    return dense_gemm


def validate_router_config(config) -> None:
    """Raise ``ValueError`` listing every setting of the Megatron ``config`` of an MoE layer that the Cake router GEMM
    does not reproduce."""
    problems = []
    if config.moe_router_dtype != "fp32":
        problems.append(f"--moe-router-dtype fp32 (got {config.moe_router_dtype!r})")
    if config.add_bias_linear:
        problems.append("no router bias (--disable-bias-linear)")
    if config.cuda_graph_impl != "none":
        problems.append(f"no CUDA graphs (--cuda-graph-impl {config.cuda_graph_impl})")
    if config.params_dtype != torch.bfloat16:
        problems.append(f"BF16 parameters (got {config.params_dtype})")
    if config.num_moe_experts % 8 or config.hidden_size % 8:
        problems.append(
            f"an expert count and a hidden size that are multiples of 8 (got {config.num_moe_experts} and "
            f"{config.hidden_size})"
        )
    if problems:
        raise ValueError("--moe-router-gemm-backend cake needs " + "; ".join(problems))


def _cake_gating(router, input: torch.Tensor) -> torch.Tensor:
    """``Router.gating`` on the Cake kernels: FP32 logits ``[..., num_experts]``."""
    if router.weight.device.type == "cpu":  # as Router.gating: the gate weight may be created on the host
        router.weight.data = router.weight.data.to(device=torch.cuda.current_device())
    return router_kernels().router_gating(input, router.weight)


def install_cake_router_gemm(model: torch.nn.Module) -> int:
    """Run the gate GEMM of every MoE router of ``model`` on the Cake kernels; returns how many routers it rebound.

    A model chunk without MoE layers (a dense pipeline stage) has none. Raises ``NotImplementedError`` for a router
    that overrides ``Router.gating``, whose computation this would replace, and ``ValueError`` for a router
    configuration the kernels do not reproduce.
    """
    from megatron.core.transformer.moe.router import Router

    routers = [(name, module) for name, module in model.named_modules() if isinstance(module, Router)]
    for name, router in routers:
        if type(router).gating is not Router.gating:
            raise NotImplementedError(
                f"--moe-router-gemm-backend cake replaces Router.gating, which {name} ({type(router).__qualname__}) "
                "overrides"
            )
        validate_router_config(router.config)
    if routers:
        router_kernels()  # fail at build time, not at the first step, when the submodule is missing
    for _, router in routers:
        router.gating = types.MethodType(_cake_gating, router)
    logger.info("--moe-router-gemm-backend cake: %d MoE routers run the Cake gate GEMM", len(routers))
    return len(routers)
