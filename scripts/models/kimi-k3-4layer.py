from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "kimi-k3", nlayers=4, rotary_base="50000")
