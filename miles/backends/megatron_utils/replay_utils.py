from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.replay_base import BaseReplayManager, RoutingReplayManager


def _register_replay_list_moe(replay_list, replay_data, models):
    layer_indices = []
    replay_idx = 0
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)

    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = replay_data[:, layer_idx]
        replay_list[replay_idx].record(layer_data)


def get_register_replay_list_func(manager: BaseReplayManager):
    if isinstance(manager, RoutingReplayManager):
        return _register_replay_list_moe
    else:
        raise ValueError(f"Unsupported manager type: {type(manager)}")
