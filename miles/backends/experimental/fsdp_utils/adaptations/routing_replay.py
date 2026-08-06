"""R3 (rollout routing replay) wiring for the FSDP backend.

The Megatron backend installs R3 inside its Megatron-LM fork: ``TopKRouter.__init__``
registers the module as a replay stream and ``topk_routing_with_score_function`` wraps
``compute_topk``. FSDP trains stock HF modeling, so the equivalent hook is installed here per
model instance -- each MoE layer's expert-selection topk is replaced by one wrapped in
``routing_replay_manager.get_topk_fn``, and the module owning it registers as a stream keyed
by that layer's global index.

Streams are keyed by *global decoder-layer index*, not by an ordinal over MoE blocks: the
rollout tensor is ``[num_tokens - 1, num_layers, topk]`` indexed by global layer id, so an
architecture with leading dense layers (GLM-4.7-Flash sets ``first_k_dense_replace=1``) would
otherwise be off by one on every layer.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)

# Matches the global decoder-layer index in a module path like "model.layers.12.mlp.gate".
# Parsing the name rather than walking a fixed attribute path keeps this agnostic to which
# wrapper transformers picked: Qwen3_5MoeForCausalLM puts decoder layers at
# model.model.layers for the text-only checkpoint and model.model.language_model.layers for
# the multimodal one.
_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass(frozen=True)
class RoutingReplayAdapter:
    """How one architecture exposes its expert-selection topk.

    ``module_cls_name`` is the class name of the module that owns the topk call and whose
    forward runs once per routing decision. It is both the discovery key and the module
    registered as a replay stream. ``install`` rebinds that module's method so the topk goes
    through the replay manager.
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
    """Find every module of class ``module_cls_name``, paired with its global layer index."""
    found: list[tuple[int, nn.Module]] = []
    for name, module in model.named_modules():
        if type(module).__name__ != module_cls_name:
            continue
        # Append "." so a module that *is* the layer's mlp ("model.layers.3.mlp") still matches.
        match = _LAYER_INDEX_RE.search(f"{name}.")
        if match is None:
            raise ValueError(
                f"cannot derive a decoder-layer index from module path {name!r}; "
                f"routing replay keys streams by global layer index"
            )
        found.append((int(match.group(1)), module))
    return sorted(found, key=lambda pair: pair[0])


def install_routing_replay(model: nn.Module, hf_config) -> int:
    """Install R3 hooks on ``model``; return the number of registered streams.

    A no-op returning 0 when routing replay is disabled, so runs without R3 keep the stock
    HF forward untouched.
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
