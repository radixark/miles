from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "minimax-m3", nlayers=4, first_k_dense_replace=2)
