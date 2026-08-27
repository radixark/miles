"""8-layer cut of GLM-5.3-Flash (layers 0-7 of the full model): 6 KDA + 2 DSA
(3, 7), 3 dense + 5 MoE -- every layer kind including the shared expert. MTP is
dropped, matching the full-model training decision."""

from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "glm5.3-flash", nlayers=8)
