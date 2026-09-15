"""Rollout routing replay (R3) for the FSDP backend.

Megatron installs R3 inside its fork's ``TopKRouter``. FSDP trains stock HF modeling, so each MoE
layer's expert-selection topk is rebound per instance (see ``models/replay_routers.py``) and
registered as a replay stream keyed by global decoder-layer index, the axis the rollout tensor
``[tokens, num_layers, topk]`` uses.

This module owns every R3 concern the actor needs: setup (``enable``, ``install``), per-rollout
data loading (``fill``), and stage control (``stage``, ``rewind``, ``reset``). The actor holds
only call sites and never touches ``routing_replay_manager`` directly.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from miles.backends.training_utils.torch_native.routing_replay import (
    FALLTHROUGH,
    RECORD,
    REPLAY_BACKWARD,
    REPLAY_FORWARD,
    enable,
    fill,
    log_prob_stage,
    reset,
    rewind,
    stage,
    uses_rollout_replay,
)
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass(frozen=True)
class RoutingReplayAdapter:
    """How one architecture exposes expert selection.

    ``module_cls_name`` names the module owning the topk call, whose forward runs once per routing
    decision; it is both the discovery key and the module registered as a replay stream.
    """

    name: str
    applies_to: Callable[[Any], bool]
    module_cls_name: str
    install: Callable[[nn.Module], None]


_ADAPTERS: list[RoutingReplayAdapter] = []


def register_routing_replay_adapter(adapter: RoutingReplayAdapter) -> None:
    _ADAPTERS.append(adapter)


def resolve_routing_replay_adapter(hf_config) -> RoutingReplayAdapter | None:
    for adapter in _ADAPTERS:
        if adapter.applies_to(hf_config):
            return adapter
    return None


def discover_moe_modules(model: nn.Module, module_cls_name: str) -> list[tuple[int, nn.Module]]:
    """Locate every ``module_cls_name`` instance with its global decoder-layer index.

    The index is parsed from the module path rather than walked to, since the layer list sits at
    different depths across HF wrappers (Qwen3.5 text-only vs multimodal).
    """
    found: list[tuple[int, nn.Module]] = []
    for name, module in model.named_modules():
        if type(module).__name__ != module_cls_name:
            continue
        match = _LAYER_INDEX_RE.search(f"{name}.")
        if match is None:
            raise ValueError(f"cannot derive a decoder-layer index from module path {name!r}")
        found.append((int(match.group(1)), module))
    return sorted(found, key=lambda pair: pair[0])


def install(model: nn.Module, hf_config) -> int:
    """Install R3 hooks on ``model`` and return the number of registered streams.

    Returns 0 without touching the model when R3 is off. Call for the actor only: a second
    registration would double ``manager.replays`` and invalidate every ``stream_idx``.
    """
    if not routing_replay_manager.enabled:
        return 0

    adapter = resolve_routing_replay_adapter(hf_config)
    if adapter is None:
        raise ValueError(
            f"no routing-replay adapter for model_type={getattr(hf_config, 'model_type', None)!r}; "
            f"rollout routing replay on the FSDP backend requires a registered adapter"
        )

    layers = discover_moe_modules(model, adapter.module_cls_name)
    if not layers:
        raise ValueError(
            f"routing-replay adapter {adapter.name!r} found no MoE layers of class "
            f"{adapter.module_cls_name!r}; the transformers version may have restructured this model"
        )

    for layer_idx, module in layers:
        adapter.install(module)
        routing_replay_manager.register_to_module(module, "routing_replay", stream_idx=layer_idx)

    logger.info(
        "[fsdp routing_replay] adapter=%s registered %d MoE layers (global indices %d..%d)",
        adapter.name,
        len(layers),
        layers[0][0],
        layers[-1][0],
    )
    return len(layers)


__all__ = [
    "FALLTHROUGH",
    "RECORD",
    "REPLAY_BACKWARD",
    "REPLAY_FORWARD",
    "enable",
    "fill",
    "install",
    "log_prob_stage",
    "reset",
    "rewind",
    "stage",
    "uses_rollout_replay",
]
