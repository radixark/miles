"""WeightBridge: registered train->rollout param-name/shape transforms for the FSDP backend."""

from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch


class ParamTransform(NamedTuple):
    matches: Callable[[str, object], bool]
    expand: Callable[[str, torch.Tensor], Iterable[tuple[str, torch.Tensor]]]


# model_type -> registered transforms, tried in registration order
_REGISTRY: dict[str, list[ParamTransform]] = {}


def register_param_transform(model_type: str, matches: Callable, expand: Callable) -> None:
    _REGISTRY.setdefault(model_type, []).append(ParamTransform(matches, expand))


def get_param_transform(name: str, param, model_type: str):
    """Return the ``expand`` fn for the transform matching this param, or None (passthrough)."""
    for transform in _REGISTRY.get(model_type, ()):
        if transform.matches(name, param):
            return transform.expand
    return None
