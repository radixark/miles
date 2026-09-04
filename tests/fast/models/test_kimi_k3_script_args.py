from __future__ import annotations

import pytest

from scripts import run_kimi_k3_lora


def _full(**kwargs):
    return run_kimi_k3_lora.ScriptArgs(model_variant="full", num_nodes=16, num_gpus_per_node=4, **kwargs)


# (pipeline, context, expected tp, expected ep) for every layout run in production.
@pytest.mark.parametrize(
    "pipeline_parallel_size,context_parallel_size,expected_tp,expected_ep",
    [(1, 1, 32, 64), (8, 1, 8, 8), (8, 2, 4, 8)],
)
def test_full_model_parallel_derivation(pipeline_parallel_size, context_parallel_size, expected_tp, expected_ep):
    args = _full(
        pipeline_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
    )
    assert args.tensor_parallel_size == expected_tp
    assert args.expert_parallel_size == expected_ep


def test_derived_ep_saturates_the_bound_post_init_validates():
    """EP must fill the non-PP ranks of one stage; TP alone under-uses it whenever CP > 1."""
    args = _full(pipeline_parallel_size=8, context_parallel_size=2)
    model_parallel = args.tensor_parallel_size * args.context_parallel_size * args.pipeline_parallel_size
    data_parallel = 64 // model_parallel
    assert args.expert_parallel_size == args.tensor_parallel_size * args.context_parallel_size * data_parallel


def test_ep_override_wins_over_derivation():
    args = _full(pipeline_parallel_size=8, context_parallel_size=2, ep_size_override=4)
    assert args.expert_parallel_size == 4


def test_tp_override_wins_and_feeds_the_ep_derivation():
    args = _full(pipeline_parallel_size=4, context_parallel_size=1, tp_size_override=8)
    assert args.tensor_parallel_size == 8
    # DP is 2 here, so the stage still holds 16 non-PP ranks.
    assert args.expert_parallel_size == 16


def test_four_layer_variant_is_unaffected():
    args = run_kimi_k3_lora.ScriptArgs(model_variant="4layer", num_nodes=1, num_gpus_per_node=8)
    assert args.tensor_parallel_size == 8
    assert args.expert_parallel_size == 8
