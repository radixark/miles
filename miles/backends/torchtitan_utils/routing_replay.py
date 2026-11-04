import contextlib
import functools
import logging

import torch.nn as nn

from miles.backends.training_utils.replay.routing_replay import FALLTHROUGH, REPLAY_BACKWARD, REPLAY_FORWARD, stage
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


def _replay_expert_selection(router: nn.Module) -> None:
    select_experts = router._select_experts

    def selected_experts(scores_TE, top_k, expert_bias_E, router_kwargs):
        return select_experts(scores_TE, expert_bias_E, **router_kwargs)

    replay_topk = routing_replay_manager.get_topk_fn(selected_experts, return_probs=False)

    def replayed_select_experts(scores_TE, expert_bias_E=None, **router_kwargs):
        return replay_topk(scores_TE, router.top_k, expert_bias_E, router_kwargs)

    router._select_experts = replayed_select_experts


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
        _replay_expert_selection(router)
        routing_replay_manager.register_to_module(router, "routing_replay", stream_idx=layer_idx)

    indices = sorted(idx for idx, _ in routers)
    logger.info(
        f"[titan routing_replay] registered {len(routers)} MoE layers " f"(global indices {indices[0]}..{indices[-1]})"
    )
    return len(routers)


def _is_installed(model_parts: list[nn.Module]) -> bool:
    return routing_replay_manager.enabled and all(
        getattr(part, _INSTALLED_ATTR, False)
        for part in model_parts  # config-access-exempt: routing replay installation marker is attached to model parts at runtime
    )


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
