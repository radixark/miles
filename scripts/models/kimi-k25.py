from model_args_utils import load_sibling_model_args


def model_args(nlayers: int = 61, first_k_dense_replace: int = 1) -> str:
    return load_sibling_model_args(
        __file__,
        "kimi-k2-thinking",
        nlayers=nlayers,
        first_k_dense_replace=first_k_dense_replace,
        beta_fast=32,
    )
