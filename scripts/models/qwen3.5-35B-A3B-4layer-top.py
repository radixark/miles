from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # TOP routes at serving's precision, so it refuses a router dtype (flag and value).
    args = load_sibling_model_args(__file__, "qwen3.5-35B-A3B-4layer").split()
    i = args.index("--moe-router-dtype")
    return " ".join(args[:i] + args[i + 2 :])
