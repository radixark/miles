from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def _is_moe_layer(config, layer_id: int) -> bool:
    if isinstance(config.moe_layer_freq, int):
        return layer_id % config.moe_layer_freq == 0
    if isinstance(config.moe_layer_freq, list):
        assert len(config.moe_layer_freq) == config.num_layers
        return config.moe_layer_freq[layer_id] != 0
    return True


def _replay_column_of_layer(config, num_columns: int) -> dict[int, int]:
    """Rollout-tensor column of each MoE layer.

    SGLang returns a column per model layer, or, for Nemotron-H, one per MoE layer in layer order.
    """
    moe_layer_ids = [layer_id for layer_id in range(config.num_layers) if _is_moe_layer(config, layer_id)]
    if num_columns == config.num_layers:
        return {layer_id: layer_id for layer_id in moe_layer_ids}
    assert num_columns == len(moe_layer_ids), (
        f"rollout routed experts have {num_columns} layers; expected {config.num_layers} (every layer) "
        f"or {len(moe_layer_ids)} (MoE layers only)"
    )
    return {layer_id: column for column, layer_id in enumerate(moe_layer_ids)}


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    """Map replay streams to Megatron MoE layers using the local model layout."""
    column_of_layer = _replay_column_of_layer(models[0].module.config, replay_data.shape[1])
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if _is_moe_layer(config, layer_id):
                layer_indices.append(layer_id)

    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = replay_data[:, column_of_layer[layer_idx]]
        replay_list[replay_idx].record(layer_data)
