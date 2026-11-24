from model_args_utils import load_sibling_model_args


def model_args(**kwargs) -> str:
    return load_sibling_model_args(__file__, "minimax-m3", rotary_base=10000000, **kwargs)
