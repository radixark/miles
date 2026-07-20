"""
Utils for megatron arguments, but not related to megatron core logic
"""


def compute_megatron_world_size_except_dp(args) -> int:
    return args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
