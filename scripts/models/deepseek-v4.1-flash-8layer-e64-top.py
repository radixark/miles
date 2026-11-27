from model_args_utils import load_sibling_model_args

# The 8-layer slice the TOP e2e test uses: DeepSeek-V4.1-Flash layers [0, 1, 2, 3, 20, 21, 24, 25] (renumbered 0-7)
# with routed experts 0-63, under TOP's model args (deepseek-v4.1-top).
_COMPRESS_RATIOS = "[0,0,2,2,1,1,1,1]"


def model_args() -> str:
    return load_sibling_model_args(
        __file__, "deepseek-v4.1-top", nlayers=8, compress_ratios=_COMPRESS_RATIOS, num_experts=64
    )
