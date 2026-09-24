"""
Utils for megatron arguments, but not related to megatron core logic
"""


def compute_megatron_world_size_except_dp(args) -> int:
    return args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size


def compute_trainer_num_cells(args, *, total_gpus: int) -> int:
    return (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1
