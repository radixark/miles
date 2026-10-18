from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # One of each layer kind: dense SWA, FP8-MoE full attention, FP8-MoE SWA, BF16-MoE SWA, dense full attention.
    # The checkpoint is pb09204048/Step-5-Preview-5layer, cut with tools/slice_step5_checkpoint.py.
    return load_sibling_model_args(__file__, "step5", layers=(0, 3, 4, 88, 91))
