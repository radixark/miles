from __future__ import annotations

import pytest

from scripts import run_kimi_k3


def _full(**kwargs):
    return run_kimi_k3.ScriptArgs(model_name="Kimi-K3", hardware="GB300", num_nodes=16, num_gpus_per_node=4, **kwargs)


def _four_layer(**kwargs):
    kwargs.setdefault("num_nodes", 1)
    return run_kimi_k3.ScriptArgs(model_name="Kimi-K3-4layer", hardware="H200", **kwargs)


# (pipeline, context, expected tp, expected ep) for every layout run in production.
@pytest.mark.parametrize(
    "pipeline_parallel_size,context_parallel_size,expected_tp,expected_ep",
    [(1, 1, 32, 64), (8, 1, 8, 8), (8, 2, 4, 8)],
)
def test_full_model_parallel_derivation(pipeline_parallel_size, context_parallel_size, expected_tp, expected_ep):
    args = _full(pipeline_parallel_size=pipeline_parallel_size, context_parallel_size=context_parallel_size)
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


def test_full_model_off_the_validated_gpu_count_needs_an_explicit_layout():
    """The 64-GPU derivation would silently produce a layout nobody has run."""
    with pytest.raises(ValueError, match="tp-size-override"):
        run_kimi_k3.ScriptArgs(model_name="Kimi-K3", hardware="H200", num_nodes=4, num_gpus_per_node=8)


@pytest.mark.parametrize("num_gpus_per_node,expected_tp", [(8, 8), (4, 4)], ids=["8-gpu-node", "4-gpu-node"])
def test_four_layer_layout_follows_the_gpu_count(num_gpus_per_node, expected_tp):
    """The rollout serves the experts replicated (EP1): Marlin is the only MXFP4 MoE runner with a LoRA path."""
    args = _four_layer(num_gpus_per_node=num_gpus_per_node)
    assert args.tensor_parallel_size == expected_tp
    assert args.expert_parallel_size == expected_tp
    assert args.rollout_tp_size == expected_tp
    assert args.rollout_ep_size == 1


def test_four_layer_rollout_tp_can_span_nodes():
    """TP16 pads the Marlin MoE intermediate, the layout the padding fix is only decidable on."""
    args = _four_layer(num_nodes=2, num_gpus_per_node=8, rollout_tp_size=16, rollout_ep_size=1)
    assert args.tensor_parallel_size == 8
    assert args.rollout_tp_size == 16
    assert args.rollout_ep_size == 1


def test_checkpoint_paths_derive_from_the_model_name():
    args = _four_layer(model_dir="/m")
    assert args.hf_checkpoint == "/m/Kimi-K3-4layer"
    assert args.bf16_checkpoint == "/m/Kimi-K3-4layer-bf16"
    assert args.ref_load == "/m/Kimi-K3-4layer-bf16_torch_dist"
