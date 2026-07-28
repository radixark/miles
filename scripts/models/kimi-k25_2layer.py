from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # Override for the 2-layer pruned debugging model (first_k_dense_replace=1):
    # 1 dense layer + 1 MoE layer. Architecture is otherwise identical to the full
    # Kimi-K2.5 / K2-Thinking, so we reuse those MODEL_ARGS and only patch the
    # layer count and the MoE-layer-frequency mask.
    return load_sibling_model_args(__file__, "kimi-k2-thinking", nlayers=2, first_k_dense_replace=1)
