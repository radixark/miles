from model_args_utils import load_sibling_model_args

# DeepSeek-V4.1 under TOP: TOP binds SequentialMLP experts behind the flex DeepEP dispatcher and builds its own V4.1
# block (--spec from top_launch), so the grouped-GEMM experts, fused permutes, the alltoall dispatcher and the V4.1
# plugin spec go. ``num_experts`` cuts the routed experts (a slice).
_TOP_REFUSED_FLAGS = ("--moe-grouped-gemm", "--moe-permute-fusion")
_TOP_REFUSED_PAIRS = ("--moe-token-dispatcher-type",)
_TOP_REFUSED_TRIPLES = ("--spec",)


def model_args(num_experts: int | None = None, **kwargs) -> str:
    args = load_sibling_model_args(__file__, "deepseek-v4.1", **kwargs).split()
    pairs = _TOP_REFUSED_PAIRS + (("--num-experts",) if num_experts is not None else ())
    kept, skip = [], 0
    for arg in args:
        if skip:
            skip -= 1
            continue
        if arg in _TOP_REFUSED_FLAGS:
            continue
        if arg in pairs:
            skip = 1
            continue
        if arg in _TOP_REFUSED_TRIPLES:
            skip = 2
            continue
        kept.append(arg)
    return " ".join(kept + (["--num-experts", str(num_experts)] if num_experts is not None else []))
