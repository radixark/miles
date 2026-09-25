import contextlib
import functools
import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from miles.backends.training_utils.replay.routing_replay import FALLTHROUGH, REPLAY_BACKWARD, REPLAY_FORWARD, stage
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _like(local: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(reference, DTensor):
        return local
    expert_dim = reference.ndim - 1
    for placement in reference.placements:
        if placement.is_shard() and placement.dim in (expert_dim, -1):
            raise RuntimeError(
                f"router scores are sharded over experts ({reference.placements}); replayed "
                "expert ids have no shard of that axis to live in"
            )
    return DTensor.from_local(local, reference.device_mesh, reference.placements)


def _token_router_forward(self, x_BLD: torch.Tensor, expert_bias_E: torch.Tensor | None = None):
    with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
        scores_BLE = self.gate(x_BLD)

    if self.score_func == "sigmoid":
        scores_BLE = torch.sigmoid(scores_BLE)
    elif self.score_func == "softmax":
        scores_BLE = F.softmax(scores_BLE, dim=-1)
    else:
        raise NotImplementedError(f"Unknown score function {self.score_func}")

    scores_for_choice_BLE = scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
    if self.num_expert_groups is not None:
        scores_for_choice_BLE = self._get_node_limited_routing_scores(scores_for_choice_BLE)

    local_choice = _local(scores_for_choice_BLE)
    b, seq_len, _ = local_choice.shape
    topk_expert_ids_BLK = _like(
        self._miles_replay_topk(local_choice.reshape(b * seq_len, -1), self.top_k).reshape(b, seq_len, self.top_k),
        scores_for_choice_BLE,
    )

    topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

    if self.route_norm:
        denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
        topk_scores_BLK = topk_scores_BLK / denominator
    topk_scores_BLK = topk_scores_BLK * self.route_scale

    return topk_scores_BLK, topk_expert_ids_BLK, scores_BLE


_INSTALLED_ATTR = "_miles_replay_installed"

_initializing: dict | None = None


def install(model_parts: list[nn.Module]) -> int:
    if not routing_replay_manager.enabled:
        return 0

    from torchtitan.models.common.moe import TokenChoiceTopKRouter

    routers: list[tuple[int, nn.Module]] = []
    for part in model_parts:
        for name, module in part.named_modules():
            if not isinstance(module, TokenChoiceTopKRouter):
                continue
            layer_key = next((p for p in name.split(".") if p.isdigit()), None)
            if layer_key is None:
                raise ValueError(f"cannot derive a decoder-layer index from router path {name!r}")
            routers.append((int(layer_key), module))

    if not routers:
        raise ValueError(
            "routing replay is enabled but this model has no torchtitan TokenChoiceTopKRouter; "
            "R3 applies to MoE models only"
        )

    for part in model_parts:
        _bracket_real_forward(part)
        setattr(part, _INSTALLED_ATTR, True)

    for layer_idx, router in sorted(routers, key=lambda pair: pair[0]):
        router._miles_replay_topk = routing_replay_manager.get_topk_fn(
            lambda scores, k: torch.topk(scores, k, dim=-1, sorted=False)[1], return_probs=False
        )
        router.forward = types.MethodType(_token_router_forward, router)
        routing_replay_manager.register_to_module(router, "routing_replay", stream_idx=layer_idx)

    indices = sorted(idx for idx, _ in routers)
    logger.info(
        f"[titan routing_replay] registered {len(routers)} MoE layers " f"(global indices {indices[0]}..{indices[-1]})"
    )
    return len(routers)


def _is_installed(model_parts: list[nn.Module]) -> bool:
    return routing_replay_manager.enabled and all(getattr(part, _INSTALLED_ATTR, False) for part in model_parts)


def bypass_schedule_initialization(model_parts: list[nn.Module]) -> None:
    global _initializing
    if not _is_installed(model_parts):
        return
    _initializing = {
        "unprobed": {id(part) for part in model_parts},
        "stage": routing_replay_manager.stage,
    }
    routing_replay_manager.stage = FALLTHROUGH


def _end_initialization() -> None:
    global _initializing
    if _initializing is None:
        return
    routing_replay_manager.stage = _initializing["stage"]
    _initializing = None


@contextlib.contextmanager
def consumption_guard(model_parts: list[nn.Module], expected: int):
    if not _is_installed(model_parts):
        yield
        return
    before = {id(replay): (replay.forward_index, replay.backward_index) for replay in routing_replay_manager.replays}
    try:
        yield
    finally:
        _end_initialization()
    for replay in routing_replay_manager.replays:
        forward_before, backward_before = before[id(replay)]
        advance = replay.forward_index - forward_before
        if advance != expected:
            raise RuntimeError(
                f"routing replay stream {replay.stream_idx} advanced {advance} times over a pass "
                f"of {expected} microbatches; the queues no longer line up with the microbatches"
            )
        recompute = replay.backward_index - backward_before
        if recompute not in (0, expected):
            raise RuntimeError(
                f"routing replay stream {replay.stream_idx} recomputed {recompute} times over a "
                f"pass of {expected} microbatches; the recompute pass is replaying the wrong "
                "microbatches"
            )


def _bracket_real_forward(part: nn.Module) -> None:
    inner = part.forward

    @functools.wraps(inner)
    def forward(*args, **kwargs):
        if _initializing is not None:
            if id(part) in _initializing["unprobed"]:
                _initializing["unprobed"].discard(id(part))
                return inner(*args, **kwargs)
            _end_initialization()
        if routing_replay_manager.stage == REPLAY_BACKWARD:
            with stage(REPLAY_FORWARD):
                return inner(*args, **kwargs)
        return inner(*args, **kwargs)

    part.forward = forward
