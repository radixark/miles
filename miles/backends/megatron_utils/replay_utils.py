from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def assign_routed_experts_streams(replays, models) -> None:
    """Give each of this rank's routing replays its stream, the decoder layer it replays.

    Megatron's routers register in layer order without their layer, and MTP routers do not register.
    """
    layer_ids = _moe_layer_ids(models)
    if len(replays) != len(layer_ids):
        raise ValueError(
            f"{len(replays)} routers registered for routing replay, but moe_layer_freq makes layers {layer_ids} "
            "of this rank MoE"
        )
    for replay, layer_id in zip(replays, layer_ids, strict=True):
        replay.stream_idx = layer_id


def _moe_layer_ids(models) -> list[int]:
    """The layers `moe_layer_freq` makes MoE among those this rank builds, over its virtual pipeline stages in order."""
    layer_ids = []
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
            layer_ids.append(layer_id)
    return layer_ids
