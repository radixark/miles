from types import SimpleNamespace

import pytest
import torch

moe_ep = pytest.importorskip("miles_plugins.top.moe_ep")


def _qwen3_30b_a3b_config(**overrides):
    values = dict(
        activation_func=torch.nn.functional.silu,
        hidden_size=2048,
        moe_ffn_hidden_size=768,
        num_moe_experts=128,
        moe_router_topk=8,
        expert_model_parallel_size=2,
        params_dtype=torch.bfloat16,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="deepep",
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_enable_expert_bias=False,
        moe_router_fusion=False,
        moe_permute_fusion=False,
        moe_apply_probs_on_input=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        moe_aux_loss_coeff=0.0,
        moe_z_loss_coeff=None,
        moe_router_force_load_balancing=False,
        moe_router_dtype=None,
    )
    return SimpleNamespace(**(values | overrides))


def test_the_qualified_configuration_is_admitted():
    moe_ep.validate_moe_config(_qwen3_30b_a3b_config(), "qwen3_moe")


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"moe_aux_loss_coeff": 1e-3}, "aux loss"),
        ({"moe_aux_loss_coeff": [0.0, 1e-3]}, "aux loss"),
        ({"moe_z_loss_coeff": 1e-3}, "z-loss"),
        ({"moe_router_force_load_balancing": True}, "force-load-balancing"),
        ({"moe_router_dtype": "fp32"}, "router-dtype"),
    ],
    ids=["aux-loss", "aux-loss-list", "z-loss", "force-load-balancing", "router-dtype"],
)
def test_router_terms_the_top_router_does_not_compute_are_refused(overrides, refusal):
    with pytest.raises(NotImplementedError, match=refusal):
        moe_ep.validate_moe_config(_qwen3_30b_a3b_config(**overrides), "qwen3_moe")
