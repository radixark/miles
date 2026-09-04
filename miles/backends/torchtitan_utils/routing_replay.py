"""Rollout routing replay (R3) for the torchtitan backend.

Training re-selects the experts the rollout engine chose, through the shared
``routing_replay_manager``: one queue of expert ids per MoE layer, keyed by
decoder-layer index. Every torchtitan MoE model routes through
``TokenChoiceTopKRouter``, so one rebound forward -- upstream's, with the
expert-selection ``torch.topk`` replaced by the manager's -- covers them all.
"""

import contextlib
import functools
import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.training_utils.replay_data import fill_replay_data, register_replay_list_sequential
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)

FALLTHROUGH = "fallthrough"
RECORD = "record"
REPLAY_FORWARD = "replay_forward"
REPLAY_BACKWARD = "replay_backward"


def uses_rollout_replay(args) -> bool:
    return bool(getattr(args, "use_rollout_routing_replay", False))


def enable(args) -> bool:
    """Settle manager state before the model is built; returns whether R3 is on."""
    routing_replay_manager.enabled = bool(getattr(args, "use_routing_replay", False))
    routing_replay_manager.enable_check_replay_result = routing_replay_manager.enabled and args.ci_test
    routing_replay_manager.register_replay_list_func = register_replay_list_sequential
    return routing_replay_manager.enabled


def _local(tensor: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _like(local: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Put ``local`` back into ``reference``'s layout; expert ids cannot be sharded over experts."""
    from torch.distributed.tensor import DTensor

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
    """torchtitan's TokenChoiceTopKRouter.forward with the expert-selection topk replaced."""
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
    """Rebind every TokenChoiceTopKRouter; returns the number of streams (0 when R3 is off).

    Call for the actor only. Streams are keyed by the router's decoder-layer
    index from the module path, which a pipeline stage keeps global.
    """
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
    """Keep the schedule's shape-inference forward and backward off the queues.

    The window ends at the first forward after every part has been probed.
    """
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
    """Assert the pass advanced every stream by exactly ``expected`` microbatches."""
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
    """Run the part's own forward on the forward cursor even inside a ``replay_backward`` step,
    leaving activation-checkpoint recompute on the backward cursor. ``functools.wraps`` keeps
    the forward signature visible to callers that introspect it."""
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


def fill(args, model_parts, data_iterators, num_microbatches, rollout_data, align=None) -> None:
    """Load the rollout's routing into the replay queues.

    Takes the iterator list because ``fill_replay_data`` resets every element.
    ``align`` reshapes each queued entry the way the trainer reshapes its input;
    padding is -1, which the manager already treats as padding.
    """
    if not uses_rollout_replay(args):
        return

    fill_replay_data(
        args=args,
        models=model_parts,
        data_iterator=data_iterators,
        num_microbatches=num_microbatches,
        rollout_data=rollout_data,
        data_key=routing_replay_manager.data_key,
        replay_list=routing_replay_manager.replays,
        register_replay_list_func=routing_replay_manager.register_replay_list_func,
        if_sp_region=routing_replay_manager.if_sp_region,
        indices_are_token_positions=routing_replay_manager.replay_indices_are_token_positions,
    )

    if align is None:
        return
    for replay in routing_replay_manager.replays:
        for i, entry in enumerate(replay.top_indices_list):
            replay.top_indices_list[i] = align(entry, -1)


def log_prob_stage(args) -> str:
    if not routing_replay_manager.enabled:
        return FALLTHROUGH
    return REPLAY_FORWARD if uses_rollout_replay(args) else RECORD


@contextlib.contextmanager
def stage(name: str):
    """Run a block with the replay manager in ``name``, restoring the previous stage after."""
    previous = routing_replay_manager.stage
    routing_replay_manager.stage = name
    try:
        yield
    finally:
        routing_replay_manager.stage = previous


def rewind() -> None:
    routing_replay_manager.clear_all_forward()


def reset() -> None:
    routing_replay_manager.clear_all()
