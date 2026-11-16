"""GPU-delta kernel backends (CUDA receiver only): the one place an SGLang kernel implementation plugs in.

Each module mirrors one kernel library's SGLang MoE runners and the quantization methods that prepare
their weights, and is named after the library; every runner and format the library serves shares its
module and helpers. To add one, write ``kernel_backends/<library>.py`` with an
``identify_layout(layer, runner_backend)`` returning layouts that satisfy ``ExpertLayout``
(``kernel_backends/base.py``), and list each exact ``moe_runner_backend`` value it serves in
``_MOE_RUNNER_BACKENDS`` with the ``moe_a2a_backend`` values admitted beside it: the only list of
admitted runners. Values match exactly, never by alias or prefix, and an unlisted runner is rejected:
there is no generic layout, because a shape-compatible view can still address the wrong bytes. A
backend module imports only ``byte_views``, ``derived`` and this package at module level, and loads
SGLang and its kernel library lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import torch

from miles.backends.sglang_utils.gpu_delta.kernel_backends import flashinfer
from miles.backends.sglang_utils.gpu_delta.kernel_backends.base import ExpertLayout


@dataclass(frozen=True)
class _RunnerRow:
    # The serving module's identify_layout, called with this row's exact runner value.
    identify: Callable[[torch.nn.Module, str], ExpertLayout]
    # The only moe_a2a_backend values admitted beside this runner.
    dispatchers: tuple[str, ...]


_MOE_RUNNER_BACKENDS = MappingProxyType(
    {
        "flashinfer_trtllm": _RunnerRow(flashinfer.identify_layout, ("none",)),
        "flashinfer_trtllm_routed": _RunnerRow(flashinfer.identify_layout, ("none",)),
        "flashinfer_mxfp4": _RunnerRow(flashinfer.identify_layout, ("none",)),
        "flashinfer_cutedsl": _RunnerRow(flashinfer.identify_layout, ("none",)),
        "flashinfer_megamoe": _RunnerRow(flashinfer.identify_layout, ("flashinfer_megamoe",)),
    }
)


def identify_expert_layout(layer, runner_backend: str) -> ExpertLayout:
    if runner_backend not in _MOE_RUNNER_BACKENDS:
        raise ValueError(
            f"unsupported GPU delta MoE runner backend: {runner_backend}; supported: {sorted(_MOE_RUNNER_BACKENDS)}"
        )
    return _MOE_RUNNER_BACKENDS[runner_backend].identify(layer, runner_backend)


def admits_moe_dispatcher(runner_backend: str, a2a_backend: str) -> bool:
    return runner_backend in _MOE_RUNNER_BACKENDS and a2a_backend in _MOE_RUNNER_BACKENDS[runner_backend].dispatchers
