"""The routed-expert GEMMs of the torchtitan models on the Cake MoE grouped GEMM of the miles-ext-kernels submodule
(``--moe-grouped-gemm-backend cake``).

:func:`convert_model_config` replaces every ``GroupedLinear.Config`` of a model's ``RoutedExperts`` (the fused
gate / up projection ``w13`` and the down projection ``w2``) with :class:`CakeGroupedLinear.Config` before the model is
built. The subclass keeps the parameter, its shape, initialisation and name, so checkpoints load unchanged; only its
``_grouped_mm`` (``torch._grouped_mm(x, w.transpose(-2, -1), offs=offsets)``) runs
``miles_ext_kernels.moe_grouped_gemm.grouped_mm`` instead: BF16 operands, FP32 accumulation, one rounding.
``--cake-moe-grouped-gemm-variant`` picks ``det`` (every reduction in a fixed order: bit-identical forward and
gradients) or ``fast`` (the same forward and input gradient, a weight gradient that may add part of its tiles in a
run-dependent order). It is not bitwise equal to ``torch._grouped_mm``.

Rejected where the model config is converted: grouped linears that are already quantized or adapted (a subclass of
``GroupedLinear``), grouped linears outside ``RoutedExperts``, and expert widths that are not multiples of 256.
CUDA-only (NVIDIA SM100a / SM103a / SM107a, torchtitan backend); the kernel package is imported only under the
``cake`` backend.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Any

import torch
from torchtitan.models.base import BaseModel
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts

logger = logging.getLogger(__name__)


def grouped_gemm_kernels():
    """The ``miles_ext_kernels.moe_grouped_gemm`` kernel package."""
    try:
        from miles_ext_kernels import moe_grouped_gemm
    except ImportError as exc:
        raise ImportError(
            "--moe-grouped-gemm-backend cake runs miles_ext_kernels.moe_grouped_gemm from the miles-ext-kernels "
            "submodule: pass --use-miles-ext-kernels and check the submodule out with "
            "'git submodule update --init 3rdparty/miles-ext-kernels'."
        ) from exc
    return moe_grouped_gemm


class CakeGroupedLinear(GroupedLinear):
    """:class:`GroupedLinear` whose grouped matmul runs the Cake MoE grouped GEMM (same parameter, same remat
    region ``<fqn>.grouped_mm``)."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def _grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        return grouped_gemm_kernels().grouped_mm(input_RI, weight_EOI.bfloat16(), offsets_E)


def _cake_config(config: GroupedLinear.Config) -> CakeGroupedLinear.Config:
    values = {
        field.name: getattr(config, field.name)  # config-access-exempt: copies every init field of the dataclass
        for field in dataclasses.fields(config)
        if field.init
    }
    return CakeGroupedLinear.Config(**values)


def convert_model_config(
    model_config: BaseModel.Config, *, backend: str, miles_ext_kernels_args: dict[str, Any] | None
) -> BaseModel.Config:
    """``model_config`` with its routed-expert grouped linears on the Cake kernels under ``backend == "cake"``, else
    unchanged. Hands the kernel package's own arguments (``miles_ext_kernels_args``) to its ``configure``."""
    if backend != "cake":
        return model_config
    kernels = grouped_gemm_kernels()
    kernels.configure(argparse.Namespace(**(miles_ext_kernels_args or {})))
    found = 0
    for fqn, config, parent, attr in model_config.traverse(GroupedLinear.Config):
        found += 1
        if isinstance(config, CakeGroupedLinear.Config):  # a config shared by several layers, already converted
            continue
        if type(config) is not GroupedLinear.Config:
            raise NotImplementedError(
                f"--moe-grouped-gemm-backend cake replaces plain GroupedLinear GEMMs; {fqn} is a "
                f"{type(config).__qualname__}"
            )
        if not isinstance(parent, RoutedExperts.Config):
            raise ValueError(
                f"--moe-grouped-gemm-backend cake converts the grouped linears of RoutedExperts; {fqn} is not one"
            )
        try:
            kernels.check_grouped_dims(config.out_features * config.num_linears, config.in_features)
        except ValueError as exc:
            raise ValueError(f"--moe-grouped-gemm-backend cake cannot run {fqn}: {exc}") from exc
        setattr(parent, attr, _cake_config(config))
    if not found:
        raise ValueError(
            "--moe-grouped-gemm-backend cake selects the MoE expert GEMMs, but the model has no MoE layer"
        )
    logger.info("--moe-grouped-gemm-backend cake: %d routed-expert grouped linears run the Cake grouped GEMM", found)
    return model_config


def prebuild(*, backend: str) -> None:
    """Compile the Cake grouped GEMM programs of the configured variant for the current device ahead of the first
    step, in parallel (each rank of a node finds the libraries the first one built)."""
    if backend != "cake":
        return
    paths = grouped_gemm_kernels().prebuild()
    logger.info("--moe-grouped-gemm-backend cake: %d grouped GEMM libraries ready", len(paths))
