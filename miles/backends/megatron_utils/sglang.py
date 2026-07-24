# the file to manage all sglang deps in the megatron actor
try:
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
except ImportError:
    per_block_cast_to_fp8 = None

# mxfp8
try:
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize
except ImportError:
    mxfp8_group_quantize = None

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

__all__ = [
    "mxfp8_group_quantize",
    "per_block_cast_to_fp8",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
]
