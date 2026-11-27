"""DeepSeek-V4.1 TOP MoE: what the trainer router refuses instead of routing with another program."""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.top.dsv41_moe import TopDSV41Router


def _config(n_hash_layers):
    return SimpleNamespace(
        moe_router_topk=6,
        num_moe_experts=16,
        hidden_size=32,
        perform_initialization=False,
        params_dtype=torch.bfloat16,
        sequence_parallel=False,
        moe_router_score_function="sqrtsoftplus",
        moe_router_enable_expert_bias=True,
        moe_n_hash_layers=n_hash_layers,
    )


def test_a_hash_layer_is_refused():
    """V4.1 routes every layer by score; a hash layer would take tid2eid, DS4's other branch."""
    with pytest.raises(NotImplementedError, match="no hash-routed layers"):
        TopDSV41Router(config=_config(n_hash_layers=2), layer_number=1)


def test_grouped_routing_is_refused():
    config = _config(n_hash_layers=0)
    config.moe_router_num_groups = 8
    with pytest.raises(NotImplementedError, match="ungrouped"):
        TopDSV41Router(config=config, layer_number=1)
