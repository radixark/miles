from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.srt.true_on_policy.registry import FAMILIES, OPS

from miles.backends.megatron_utils.arguments import admit_true_on_policy_trainer
from miles_plugins.top.registry import BUILDERS, TRAINER_CELLS, TrainerLayout, admit_trainer


def _layout(**kwargs):
    fields = dict(tp=1, etp=1, ep=1, cp=1, pp=1, virtual_pipeline=False, sequence_parallel=False, allgather_cp=False)
    fields.update(kwargs)
    return TrainerLayout(**fields)


def test_trainer_cells_cover_every_op_and_family():
    """An op or family added to SGLang's registry without its trainer half would only fail on a GPU launch."""
    assert set(TRAINER_CELLS) == {op.name for op in OPS if op.in_trainer}
    assert set(BUILDERS) == {family.name for family in FAMILIES}


@pytest.mark.parametrize(
    ("layout", "refusal"),
    [
        (dict(tp=2, etp=2), "TP and expert TP must be 1"),
        (dict(etp=2), "TP and expert TP must be 1"),
        (dict(sequence_parallel=True), "sequence parallelism"),
        (dict(pp=2, virtual_pipeline=True), "virtual pipeline"),
        (dict(cp=2), "all-gather form only"),
    ],
)
def test_core_trainer_refusals(layout, refusal):
    with pytest.raises(NotImplementedError, match=refusal):
        admit_trainer("not_a_top_model", _layout(**layout), fp8=False)


def test_a_block_fp8_checkpoint_of_a_family_without_fp8_ops_is_refused():
    """Admission refuses an FP8 checkpoint of a family with no block-FP8 ops instead of running its BF16 ops."""
    admit_trainer("qwen3", _layout(), fp8=False)
    with pytest.raises(NotImplementedError, match="Qwen3 dense has no block-FP8 program"):
        admit_trainer("qwen3", _layout(), fp8=True)


@pytest.mark.parametrize("ep", [1, 2, 4])
def test_qwen3_moe_is_refused_at_every_layout(ep):
    with pytest.raises(NotImplementedError, match="op qwen3_moe_experts: ep=.* refused: no e2e gate runs Qwen3 MoE"):
        admit_trainer("qwen3_moe", _layout(ep=ep), fp8=False)


def test_images_are_admitted_where_the_family_has_a_vision_program():
    """A run with images adds the tower's cells to the text program's: the Qwen GDN families train images wherever
    they train text (all-gather CP included), and a family without a vision program refuses images outright."""
    cp2 = _layout(cp=2, allgather_cp=True)
    assert admit_trainer("qwen3_5", cp2, fp8=False, vision=True).name == "Qwen GDN dense"
    assert admit_trainer("qwen3_5_moe", _layout(ep=4, cp=2, allgather_cp=True), fp8=False, vision=True)
    with pytest.raises(NotImplementedError, match="Qwen3 dense has no vision program"):
        admit_trainer("qwen3", _layout(), fp8=False, vision=True)


def test_all_gather_cp_passes_the_core_rules():
    with pytest.raises(NotImplementedError, match="no TOP family admits"):
        admit_trainer("not_a_top_model", _layout(cp=2, allgather_cp=True), fp8=False)


def test_true_on_policy_mode_refuses_an_unadmitted_model_before_launch():
    args = SimpleNamespace(
        true_on_policy_mode=True,
        hf_checkpoint="/ckpt",
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
        allgather_cp=False,
        use_session_server=False,
        multimodal_keys=None,
    )
    with (
        patch(
            "miles.backends.megatron_utils.arguments.load_hf_config",
            return_value=SimpleNamespace(model_type="not_a_top_model"),
        ),
        patch("miles_plugins.top.fp8_config.checkpoint_is_block_fp8", return_value=False),
    ):
        with pytest.raises(NotImplementedError, match="no TOP family admits model_type 'not_a_top_model'"):
            admit_true_on_policy_trainer(args)


def test_qwen_gdn_refuses_session_rollouts_before_launch():
    """No gate runs session rollouts under GDN TOP."""
    args = SimpleNamespace(true_on_policy_mode=True, hf_checkpoint="/ckpt", use_session_server=True)
    with patch("miles.true_on_policy.gdn.is_top_gdn", return_value=True):
        with pytest.raises(NotImplementedError, match="session rollouts"):
            admit_true_on_policy_trainer(args)


def test_spec_build_refuses_an_unadmitted_model_before_pinning_serving():
    """The --spec path is the binding admission: it must refuse even when the launch-time check was bypassed."""
    from miles_plugins.top import spec

    config = SimpleNamespace(
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=None,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )
    with (
        patch.object(spec, "checkpoint_is_block_fp8", return_value=False),
        patch.object(spec, "hf_model_type", return_value="not_a_top_model"),
        patch.object(spec, "pin_sglang") as pin,
        pytest.raises(NotImplementedError, match="no TOP family admits"),
    ):
        spec.get_top_spec(
            SimpleNamespace(true_on_policy_mode=True, allgather_cp=False, hf_checkpoint="/ckpt", multimodal_keys=None),
            config,
            vp_stage=None,
        )
    pin.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        (dict(recompute_granularity="selective"), "selective activation recompute"),
        (dict(recompute_granularity="full", fp8="e4m3"), "te_checkpoint"),
        (dict(overlap_moe_expert_parallel_comm=True), "combined 1F1B"),
        (dict(cuda_graph_impl="local"), "CUDA-graph backward"),
    ],
)
def test_backward_paths_the_patches_miss_are_refused(overrides, refusal):
    """These paths run backward or recompute outside the patched hooks, so their bytes would differ from serving."""
    from miles_plugins.top.install import refuse_unpatched_backward

    fields = dict(
        recompute_granularity=None,
        fp8=None,
        fp4=None,
        overlap_moe_expert_parallel_comm=False,
        cuda_graph_impl="none",
        external_cuda_graph=False,
    )
    fields.update(overrides)
    with pytest.raises(NotImplementedError, match=refusal):
        refuse_unpatched_backward(SimpleNamespace(**fields))
