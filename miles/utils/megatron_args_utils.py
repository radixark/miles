"""
Utils for megatron arguments, but not related to megatron core logic
"""

import itertools


def compute_megatron_world_size_except_dp(args) -> int:
    return args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size


def compute_trainer_num_cells(args, *, total_gpus: int) -> int:
    return (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1


def pp_layer_splits(args) -> list[int]:
    """The layers where Megatron's pipeline model chunks meet, ascending, without 0 and ``num_layers``.

    Each pipeline stage builds its decoder layers as ``virtual_pipeline_model_parallel_size`` chunks
    (one without VPP), interleaved: chunk 0 of every stage in stage order, then chunk 1, and so on.
    The first and last stages hold ``decoder_first/last_pipeline_num_layers`` when set, and the other
    stages split the rest evenly. Every chunk is one ``[split, next split)`` range, so a stage that
    reads only the ranges of its chunks reads only its own layers. [] with a single stage.
    """
    pp = args.pipeline_model_parallel_size
    if pp == 1:
        return []
    vp = args.virtual_pipeline_model_parallel_size or 1
    first, last = args.decoder_first_pipeline_num_layers, args.decoder_last_pipeline_num_layers
    even_stages = pp - (first is not None) - (last is not None)
    even = (args.num_layers - (first or 0) - (last or 0)) // even_stages if even_stages else 0
    stage_layers = [even] * pp
    if first is not None:
        stage_layers[0] = first
    if last is not None:
        stage_layers[-1] = last
    chunk_ends = itertools.accumulate(layers // vp for _ in range(vp) for layers in stage_layers)
    return sorted(set(chunk_ends) - {0, args.num_layers})
