import itertools
from types import SimpleNamespace

import pytest

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.trainer_utils import compute_trainer_total_gpus
from miles.utils.megatron_args_utils import compute_trainer_num_cells, pp_layer_splits


@pytest.mark.parametrize("role,total_gpus,independent_cells", [("actor", 16, 2), ("critic", 24, 3)])
@pytest.mark.parametrize("independent_dp", [False, True])
def test_independent_cells_include_tensor_pipeline_and_context_parallelism(
    role: str, total_gpus: int, independent_cells: int, independent_dp: bool
) -> None:
    """Every model-parallel dimension belongs to a cell, using each trainer role's GPU count."""
    scaling = ScalingConfig.model_construct(
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        critic_num_nodes=3,
        critic_num_gpus_per_node=8,
    )
    args = SimpleNamespace(
        indep_dp=independent_dp,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
    )

    actual_gpus = compute_trainer_total_gpus(scaling, role=role)
    actual_cells = compute_trainer_num_cells(args, total_gpus=actual_gpus)

    assert actual_gpus == total_gpus
    assert actual_cells == (independent_cells if independent_dp else 1)


@pytest.mark.parametrize(
    "num_layers,pp,vp,first,last,splits",
    [
        (4, 1, None, None, None, []),
        (48, 4, None, None, None, [12, 24, 36]),
        # Interleaved: every stage's chunks meet the other stages' at the same edges.
        (48, 2, 2, None, None, [12, 24, 36]),
        (8, 2, 4, None, None, [1, 2, 3, 4, 5, 6, 7]),
        (5, 2, None, 3, None, [3]),
        (61, 4, None, 16, None, [16, 31, 46]),
        (30, 3, None, None, 6, [12, 24]),
        (61, 4, None, 16, 15, [16, 31, 46]),
        (28, 4, 2, 4, 4, [2, 7, 12, 14, 16, 21, 26]),
    ],
)
def test_pp_layer_splits_put_every_megatron_model_chunk_between_two_splits(num_layers, pp, vp, first, last, splits):
    """Each pipeline rank's chunk ``[offset, offset + n)`` (Megatron's own assignment) is one range of the splits."""
    get_num_layers_to_build = pytest.importorskip(
        "megatron.core.transformer.transformer_block"
    ).get_num_layers_to_build
    get_transformer_layer_offset = pytest.importorskip(
        "megatron.core.transformer.transformer_layer"
    ).get_transformer_layer_offset
    args = SimpleNamespace(
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
        num_layers=num_layers,
        decoder_first_pipeline_num_layers=first,
        decoder_last_pipeline_num_layers=last,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
        num_layers=num_layers,
        num_layers_in_first_pipeline_stage=first,
        num_layers_in_last_pipeline_stage=last,
        pipeline_model_parallel_layout=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
    )

    actual = pp_layer_splits(args)

    assert actual == splits
    ranges = set(itertools.pairwise([0, *actual, num_layers]))
    for pp_rank, vp_stage in itertools.product(range(pp), range(vp) if vp else [None]):
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
        chunk = (offset, offset + get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank))
        assert chunk in ranges, (pp_rank, vp_stage, chunk, actual)
