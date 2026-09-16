from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "qwen3-30B-A3B", nlayers=5)
