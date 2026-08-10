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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from miles.backends.training_utils.replay_data import fill_replay_data, register_replay_list_sequential
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")

FALLTHROUGH = "fallthrough"
RECORD = "record"
REPLAY_FORWARD = "replay_forward"
REPLAY_BACKWARD = "replay_backward"


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


def uses_rollout_replay(args) -> bool:
    """True when routing comes from the rollout rather than from a recording pass."""
    return bool(getattr(args, "use_rollout_routing_replay", False))


def enable(args) -> bool:
    """Settle manager state before the model is built, and report whether R3 is on.

    ``--use-rollout-routing-replay`` sets ``use_routing_replay`` during arg validation;
    ``--use-routing-replay`` alone selects the record-then-replay variant.
    """
    routing_replay_manager.enabled = bool(getattr(args, "use_routing_replay", False))
    routing_replay_manager.enable_check_replay_result = routing_replay_manager.enabled and args.ci_test
    routing_replay_manager.register_replay_list_func = register_replay_list_sequential
    return routing_replay_manager.enabled


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


def fill(args, model, data_iterator, num_microbatches, rollout_data) -> None:
    """Load the rollout's routing into the per-layer replay queues.

    Takes the iterator list rather than a single iterator: ``fill_replay_data`` resets every
    element and reads through element 0, so call before the caller unwraps it.
    """
    if not uses_rollout_replay(args):
        return

    fill_replay_data(
        args=args,
        models=model,
        data_iterator=data_iterator,
        num_microbatches=num_microbatches,
        rollout_data=rollout_data,
        data_key=routing_replay_manager.data_key,
        replay_list=routing_replay_manager.replays,
        register_replay_list_func=routing_replay_manager.register_replay_list_func,
        if_sp_region=routing_replay_manager.if_sp_region,
        indices_are_token_positions=routing_replay_manager.replay_indices_are_token_positions,
    )


def log_prob_stage(args) -> str:
    """Stage for the actor log-prob pass.

    Rollout replay consumes the queues filled from the rollout; the record-then-replay variant
    has nothing to consume yet and records this pass instead.
    """
    if not routing_replay_manager.enabled:
        return FALLTHROUGH
    return REPLAY_FORWARD if uses_rollout_replay(args) else RECORD


@contextmanager
def stage(name: str):
    """Run a block with the replay manager in ``name``, restoring the previous stage after.

    Nesting a ``replay_forward`` forward inside a ``replay_backward`` step is what lets
    activation-checkpoint recompute draw from the independent backward cursor.
    """
    previous = routing_replay_manager.stage
    routing_replay_manager.stage = name
    try:
        yield
    finally:
        routing_replay_manager.stage = previous


def rewind() -> None:
    """Return the forward cursors to the head of their queues."""
    routing_replay_manager.clear_all_forward()


def reset() -> None:
    """Drop the recorded routing once the rollout is done training."""
    routing_replay_manager.clear_all()
