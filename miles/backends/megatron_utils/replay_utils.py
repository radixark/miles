from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.backends.training_utils.replay.data import locate_stream, stream_blocks


def register_replay_list_moe(replay_list, replay_data, *, models, blocks, **_kwargs):
    """Map replay streams to Megatron MoE layers using the local model layout.

    `replay_data[b]` holds layers `blocks[b]`.
    """
    layer_indices = []
    for (offset, end), model in zip(vp_stage_layers(models), models, strict=True):
        config = model.module.config
        for layer_id in range(offset, end):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)

    for replay_idx, layer_idx in enumerate(layer_indices):
        block, column = locate_stream(blocks, layer_idx)
        replay_list[replay_idx].record(replay_data[block][:, column])


def vp_stage_layers(models) -> list[tuple[int, int]]:
    """The decoder layers `[offset, offset + n)` each virtual pipeline stage of this rank builds."""
    layers = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        layers.append((offset, offset + get_num_layers_to_build(config, vp_stage=vp_stage)))
    return layers


def replay_blocks(manager, models) -> list[tuple[int, int]]:
    """The stream ranges `manager` registers on this pipeline rank, for `attach_replay_data`.

    Routed experts: each virtual pipeline stage's layers; other replays: the runs of their streams. Each is one part
    of the stored field, which SGLang splits at every stage's edges (`pp_layer_splits`), so a rank reads only its own.
    """
    if manager.register_replay_list_func is register_replay_list_moe:
        return vp_stage_layers(models)
    return stream_blocks(manager.replays)
