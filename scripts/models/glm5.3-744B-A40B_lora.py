from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # GLM-5.3-BF16 has the same architecture and DSA index-sharing schedule as GLM-5.2.
    return load_sibling_model_args(__file__, "glm5.2-744B-A40B_lora")
