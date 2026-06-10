from argparse import Namespace

import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-fast")


def test_qwen3moe_converter_accepts_explicit_true_on_policy_layernorm_names():
    from miles.backends.megatron_utils.megatron_to_hf.qwen3moe import convert_qwen3moe_to_hf

    args = Namespace(
        hidden_size=4,
        kv_channels=2,
        num_attention_heads=2,
        num_query_groups=1,
    )
    param = torch.ones(4)

    assert convert_qwen3moe_to_hf(
        args,
        "module.module.decoder.layers.0.input_layernorm.weight",
        param,
    ) == [("model.layers.0.input_layernorm.weight", param)]
    assert convert_qwen3moe_to_hf(
        args,
        "module.module.decoder.layers.0.pre_mlp_layernorm.weight",
        param,
    ) == [("model.layers.0.post_attention_layernorm.weight", param)]
