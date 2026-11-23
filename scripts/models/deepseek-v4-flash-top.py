from model_args_utils import load_sibling_model_args

# TOP binds SequentialMLP experts behind the unfused flex dispatcher; it refuses grouped-GEMM experts and fused permutes.
_TOP_REFUSED = ("--moe-grouped-gemm", "--moe-permute-fusion")


def model_args() -> str:
    args = load_sibling_model_args(__file__, "deepseek-v4-flash").split()
    return " ".join(arg for arg in args if arg not in _TOP_REFUSED)
