"""A shrunk Inkling2 checkpoint config and the model builders the Inkling2 tests share."""

import json
from argparse import Namespace

import torch

# the released text_config's layout, shrunk: layers 2 and 5 are global, the loop core is layers 1-3
TEXT_CONFIG = dict(
    model_max_length=4096,
    hidden_size=64,
    num_hidden_layers=6,
    vocab_size=128,
    unpadded_vocab_size=120,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=16,
    d_rel=4,
    rel_extent=32,
    q_bias=False,
    o_bias=False,
    log_scaling_n_floor=16,
    log_scaling_alpha=0.1,
    rms_norm_eps=1e-6,
    use_value_norm_no_gamma=True,
    qk_l2_clip_threshold=22.627416997969522,
    rel_l2_clip_threshold=8.0,
    local_layer_ids=[0, 1, 3, 4],
    sconv_kernel_size=4,
    final_logit_softcapping=30.0,
    sliding_window_size=8,
    n_layers_per_attn_res_block=2,
    attn_res_normalize_values=True,
    n_routed_experts=4,
    num_experts_per_tok=2,
    n_shared_experts=0,
    intermediate_size=32,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    route_scale=8.0,
    use_gate_bias=True,
    loop_config=dict(core_start=1, core_end=4, num_passes=3, gate_rank=4, is_varloop=False),
)


def write_checkpoint_config(directory, *, model_type="inkling2_loop", **text_overrides) -> str:
    config = {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": model_type,
        "text_config": {**TEXT_CONFIG, **text_overrides},
    }
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


def model_config(directory, *, seq_length=256, model_type="inkling2_loop", **text_overrides):
    from torchtitan.config.parallelism import ParallelismConfig

    from miles.backends.torchtitan_utils.config import build_model_config

    checkpoint = write_checkpoint_config(directory, model_type=model_type, **text_overrides)
    return build_model_config(
        Namespace(backend=Namespace(titan_model_name="inkling2", titan_model_flavor="tiny", seq_length=seq_length)),
        parallelism=ParallelismConfig(),
        hf_assets_path=checkpoint,
    )


def random_model(directory):
    torch.manual_seed(0)
    config = model_config(directory)
    model = config.build()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.1)
    return config, model.eval()


def as_trained(model):
    """The dtypes FSDP's mixed-precision policies give the forward: bf16, with the residual queries in fp32."""
    model = model.to(torch.bfloat16)
    queries = [model.output_query, model.loop_queries]
    for block in model.layers.values():
        queries += [block.mlp_query, block.attn_query]
    for module in queries:
        if module is not None:
            module.float()
    return model
