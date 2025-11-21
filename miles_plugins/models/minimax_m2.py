import torch.distributed.nn
from dataclasses import dataclass
from types import MethodType

import torch.nn
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build


def get_qwen3_next_spec(args, config, vp_stage):
    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    for layer_id in range(num_layers_to_build):
        layer_spec = transformer_layer_spec.layer_specs[layer_id]
        self_attention_submodules = layer_spec.submodules.self_attention.submodules
        assert isinstance(self_attention_submodules, SelfAttentionSubmodules)
        for k, num_heads in [
            ("q_layernorm", args.num_attention_heads),
            ("k_layernorm", args.num_query_groups),
        ]:
            v_old = getattr(self_attention_submodules, k)
            v_new = _create_per_layer_rms_norm(v_old, num_heads=num_heads)
            setattr(self_attention_submodules, k, v_new)

    return transformer_layer_spec


def _create_per_layer_rms_norm(inner_cls: type, num_heads: int) -> type:
    tp_group = TODO
    tp_group_rank = torch.distributed.get_rank(tp_group)
    tp_group_world_size = torch.distributed.get_world_size(tp_group)

    if tp_group_world_size == 1:
        return inner_cls

    return ModuleSpec(
        module=_PerLayerRMSNorm,
        params=dict(
            extra=_PerLayerRMSNormExtraArgs(
                inner_cls=inner_cls,
                num_heads=num_heads,
                tp_group=tp_group,
                tp_group_rank=tp_group_rank,
                tp_group_world_size=tp_group_world_size,
            )
        ),
    )


@dataclass
class _PerLayerRMSNormExtraArgs:
    inner_cls: type
    num_heads: int
    tp_group: any
    tp_group_rank: int
    tp_group_world_size: int


# ref: WrappedTorchNorm, TENorm
class _PerLayerRMSNorm:
    def __new__(cls, *args, hidden_size: int, extra: _PerLayerRMSNormExtraArgs, **kwargs):
        # MiniMax-M2 head_dim, can remove this assertion when more model is to be supported
        assert hidden_size == 128

        obj = extra.inner_cls(*args, hidden_size=hidden_size * extra.num_heads, **kwargs)

        original_forward = obj.forward

        # Slow, should optimize later
        def _modified_forward(x: torch.Tensor) -> torch.Tensor:
            original_shape = x.shape

            x = _all_gather(x, group=extra.tp_group, concat_dim=-1)
            assert x.shape[-1] == hidden_size * extra.num_heads

            x = original_forward(x)

            x = _slice(x, dim=-1, rank=extra.tp_group_rank, num_slices=extra.tp_group_world_size)
            assert x.shape == original_shape

            return x

        obj.forward = MethodType(_modified_forward, obj)

        return obj


def _all_gather(x, *, group, concat_dim: int):
    out_list = torch.distributed.nn.all_gather(x, group=group)
    return torch.concat(out_list, dim=concat_dim)


def _slice(x, *, dim: int, rank: int, num_slices: int):
    tensor_len = x.shape[dim]

    slice_len, remainder = divmod(tensor_len, num_slices)
    assert remainder == 0

    return torch.narrow(x, dim=dim, start=slice_len * rank, length=slice_len)
