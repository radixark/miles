from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def _live_moe_replays(models):
    """Replay buffers of the MoE routers that are actually part of ``models``.

    ``register_to_module`` appends to the manager's global replay list at router
    construction time, so a router that is built and then discarded still leaves
    its replay behind (e.g. a model provider that rebuilds the decoder after
    ``GPTModel.__init__`` has already built one). Filling by list position then
    lands recorded streams on dead routers while every live router keeps an
    empty buffer, which surfaces later as ``pop_forward out of range`` in the
    middle of the first forward. Binding by module identity keeps each stream on
    the router that will actually run.
    """
    return [
        replay
        for model in models
        for _, module in model.named_modules()
        if (replay := getattr(module, "routing_replay", None)) is not None
    ]


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    """Map replay streams to Megatron MoE layers using the local model layout."""
    layer_indices = []
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

    replays = _live_moe_replays(models)
    assert len(replays) == len(layer_indices), (
        f"routing replay: {len(replays)} live MoE routers on this rank but "
        f"{len(layer_indices)} replay streams to fill"
    )
    for replay, layer_idx in zip(replays, layer_indices, strict=True):
        replay.record(replay_data[:, layer_idx])
