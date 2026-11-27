from model_args_utils import load_sibling_model_args

# TOP binds SequentialMLP experts behind the unfused flex dispatcher and routes at serving's precision: it refuses
# grouped-GEMM experts and a router dtype, and turns permute fusion off.
_TOP_DROPPED = ("--moe-grouped-gemm", "--moe-permute-fusion")


def model_args() -> str:
    args = load_sibling_model_args(__file__, "glm5.2-744B-A40B_5layer").split()
    i = args.index("--moe-router-dtype")
    return " ".join(arg for arg in args[:i] + args[i + 2 :] if arg not in _TOP_DROPPED)
