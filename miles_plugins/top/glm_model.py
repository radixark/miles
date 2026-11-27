"""Local projection support for the GLM TOP spec."""

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.post_training.modelopt.layers import Linear


def match_linear_dtype(x, linear):
    """Match projection operand dtypes for explicitly bound TOP GLM calls."""
    if getattr(linear, "is_top_fp8", False):  # config-access-exempt: only TOP FP8 modules carry is_top_fp8
        return x
    weight = getattr(
        linear, "weight", None
    )  # config-access-exempt: module variants differ in whether they define weight
    if weight is not None and x.dtype != weight.dtype:
        return x.to(weight.dtype)
    return x


class GLMLocalSpecProvider(LocalSpecProvider):
    """Add replicated projections needed by MLA and the DSA indexer."""

    def linear(self) -> type:
        return Linear
