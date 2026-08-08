import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)


def strip_param_name_prefix(name: str | None) -> str | None:
    if name is None:
        return None
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name


def report_nonfinite_grads(
    named_params: Iterable[tuple[str, torch.nn.Parameter]], header: str, limit: int = 12
) -> None:
    """Log which of the given params' grads are non-finite (and how many are NaN), plus the ones that stayed finite.

    Worth the walk: a non-finite grad norm is otherwise a bare ``train/grad_norm = nan``
    alongside a perfectly finite loss, which says nothing about where it came from. The
    finite minority is what localises the fault: gradients flow from the loss down, so the
    shallowest clean tensor sits just above the layer that first went non-finite.
    """
    bad: list[str] = []
    good: list[str] = []
    n_nan = 0
    for name, param in named_params:
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is None:
            continue
        if torch.isfinite(grad).all():
            good.append(name)
            continue
        bad.append(name)
        n_nan += int(torch.isnan(grad).any())
    total = len(bad) + len(good)
    logger.error(
        f"{header}: {len(bad)}/{total} grads are non-finite, {n_nan} of them NaN; "
        f"first {limit} non-finite: {bad[:limit]}; all {len(good)} finite: {good[:64]}"
    )
