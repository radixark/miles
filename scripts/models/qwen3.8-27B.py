from model_args_utils import load_sibling_model_args


def model_args() -> str:
    """Qwen3.8-27B ships the same config.json as Qwen3.5-27B, down to every shape and gate."""
    return load_sibling_model_args(__file__, "qwen3.5-27B")
