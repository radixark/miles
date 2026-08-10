from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # Override for the 6-layer pruned GLM-5.1 toy (jybsuper/GLM-5.1-6layer):
    # first 6 layers = 3 dense + 3 MoE.
    return load_sibling_model_args(__file__, "glm5.1-744B-A40B_lora", n_moe_layers=3)
