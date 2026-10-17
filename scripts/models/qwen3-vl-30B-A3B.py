from model_args_utils import load_sibling_model_args


def model_args(**kwargs) -> str:
    # Qwen3-VL shares the text MoE dimensions, but ties its embeddings and uses a different RoPE base.
    args = load_sibling_model_args(__file__, "qwen3-30B-A3B", **{"rotary_base": "5000000", **kwargs})
    return args.replace("--untie-embeddings-and-output-weights", "")
