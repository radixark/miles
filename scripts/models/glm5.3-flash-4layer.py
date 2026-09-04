from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "glm5.3-flash", nlayers=4, first_k_dense_replace=1)
