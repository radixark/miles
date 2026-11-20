from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # The first 8 layers and 64 experts: SWA layers 0-4/6/7, the full-attention layer 5,
    # and the attention-residual block boundary at layer 6.
    return load_sibling_model_args(__file__, "inkling2", num_layers=8, num_experts=64)
