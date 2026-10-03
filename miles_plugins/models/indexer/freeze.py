"""Freezing the indexer.

No RL recipe trains the indexer, and leaving it trainable is not neutral: it
collects optimizer state, and decoupled weight decay shifts the selection every
step with no learning signal behind it. Freezing runs before the optimizer is
built, so Megatron allocates no gradient buffer and no Adam state for it.
"""

import logging
from fnmatch import fnmatch

import torch.nn as nn

logger = logging.getLogger(__name__)


def _matching_parameter_names(module: nn.Module, param_globs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        name for name, _ in module.named_parameters() if any(fnmatch(name, pattern) for pattern in param_globs)
    )


def freeze_indexer_parameters(
    module: nn.Module,
    param_globs: tuple[str, ...],
    *,
    expect_match: bool = True,
) -> tuple[str, ...]:
    """Set `requires_grad=False` on every parameter of `module` matching `param_globs`.

    Raises when the globs match nothing, because that means the indexer kept
    training silently. A pipeline stage holding no indexer layer legitimately
    matches nothing and passes `expect_match=False`.
    """
    frozen_names = _matching_parameter_names(module, param_globs)

    if not frozen_names:
        if expect_match:
            raise ValueError(
                f"Indexer param_globs {param_globs} matched no parameter of "
                f"{type(module).__name__}, so the indexer would silently train. "
                f"Parameters present: {[name for name, _ in module.named_parameters()]}"
            )
        return ()

    frozen_lookup = set(frozen_names)
    for name, parameter in module.named_parameters():
        if name in frozen_lookup:
            parameter.requires_grad = False

    logger.debug("Froze %d indexer parameters on %s", len(frozen_names), type(module).__name__)
    return frozen_names
