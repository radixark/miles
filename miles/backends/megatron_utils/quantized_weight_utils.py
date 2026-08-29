"""Wiring for --clear-quantized-weight-workspaces-on-offload. Drops TransformerEngine's
cached quantized weights so offload does not back them up."""

import logging
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)


def clear_quantized_weight_workspaces(models: Sequence[torch.nn.Module]) -> int:
    """Under a low-precision recipe TE caches the quantized weight of every module in
    ``_fp8_workspaces``. It is derived from the high-precision weight, so TE rebuilds it on the
    next forward via the cache-miss path.

    Skipped under CUDA graphs: a captured graph replays with the workspace address baked in, so
    freeing it would let a later allocation reuse that memory.
    """
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    if any(model_chunk.config.cuda_graph_impl != "none" for model_chunk in models):
        logger.info("Keeping cached quantized weight workspaces: clearing them is unsafe with CUDA graphs")
        return 0

    num_cleared = 0
    for model_chunk in models:
        for module in model_chunk.modules():
            if isinstance(module, TransformerEngineBaseModule):
                num_cleared += len(module._fp8_workspaces)
                module._fp8_workspaces.clear()
    return num_cleared
