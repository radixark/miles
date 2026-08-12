from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "qwen3-4B", rotary_base=5000000)
