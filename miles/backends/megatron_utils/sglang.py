# the file to manage all sglang deps in the megatron actor
import inspect

try:
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except ImportError:
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None
    should_deepgemm_weight_requant_ue8m0 = None


def _accept_grouped_moe(should_requant):
    # sgl-project/sglang#43556 (v0.5.23) added `grouped_moe`; the ROCm image still runs
    # sglang-miles-v0.5.22-final. Drop this once the ROCm image is on v0.5.23.
    if "grouped_moe" in inspect.signature(should_requant).parameters:
        return should_requant

    def should_requant_without_grouped_moe(*, weight_block_size, grouped_moe=False):
        return should_requant(weight_block_size=weight_block_size)

    return should_requant_without_grouped_moe


if should_deepgemm_weight_requant_ue8m0 is not None:
    should_deepgemm_weight_requant_ue8m0 = _accept_grouped_moe(should_deepgemm_weight_requant_ue8m0)

try:
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
except ImportError:
    per_block_cast_to_fp8 = None

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

__all__ = [
    "is_layer_skipped",
    "per_block_cast_to_fp8",
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
]
