import re

from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.replay_base import BaseReplayManager, RoutingReplayManager



_DECODER_LAYER_NAME_RE = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)")


def _iter_local_routing_replays(models):
    seen_replays = set()
    for vp_stage, model in enumerate(models):
        module = getattr(model, "module", model)
        config = module.config
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

        for module_name, submodule in module.named_modules():
            replay = getattr(submodule, "routing_replay", None)
            if replay is None or id(replay) in seen_replays:
                continue

            match = _DECODER_LAYER_NAME_RE.search(module_name)
            if match is None:
                raise ValueError(
                    f"Found routing replay attached to an unexpected module path: {module_name!r}. "
                    "Expected a decoder.layers.<idx> path."
                )

            seen_replays.add(id(replay))
            local_layer_idx = int(match.group(1))
            yield offset + local_layer_idx, replay


def _register_replay_list_moe(replay_list, replay_data, models):
    del replay_list

    if replay_data.ndim != 3:
        raise ValueError(f"Expected replay_data to be 3D [tokens, layers, topk], got shape {tuple(replay_data.shape)}")

    local_routing_replays = list(_iter_local_routing_replays(models))
    if not local_routing_replays:
        return

    num_layers = replay_data.shape[1]
    for layer_idx, replay in local_routing_replays:
        if layer_idx >= num_layers:
            raise IndexError(
                f"Replay layer index {layer_idx} is out of bounds for replay_data with {num_layers} layers."
            )
        layer_data = replay_data[:, layer_idx]
        replay.record(layer_data)


def get_register_replay_list_func(manager: BaseReplayManager):
    if isinstance(manager, RoutingReplayManager):
        return _register_replay_list_moe
    else:
        raise ValueError(f"Unsupported manager type: {type(manager)}")
