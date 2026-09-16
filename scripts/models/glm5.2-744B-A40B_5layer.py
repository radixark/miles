from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # Override for 5-layer pruned model (first 5 layers: 3 dense + 2 MoE).
    # Keeps at least one computing + one skip layer so the DSA cross-layer index
    # sharing path is exercised (computing layers 0,1,2; skip layers 3,4).
    return load_sibling_model_args(__file__, "glm5.2-744B-A40B", n_moe_layers=2)
