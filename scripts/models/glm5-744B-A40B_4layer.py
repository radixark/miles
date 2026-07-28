from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # Override for 4-layer pruned model (first 4 layers: 3 dense + 1 MoE)
    return load_sibling_model_args(__file__, "glm5-744B-A40B", n_moe_layers=1)
