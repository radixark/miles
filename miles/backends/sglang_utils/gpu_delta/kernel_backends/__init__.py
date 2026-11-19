"""Map each supported SGLang MoE runner to its FlashInfer weight layout and dispatcher.

The routed and MXFP4 TRTLLM runners use the same storage family as the ordinary TRTLLM runner.
Additional kernel libraries must identify their own layouts before admission.
"""

from miles.backends.sglang_utils.gpu_delta.kernel_backends import flashinfer
from miles.backends.sglang_utils.gpu_delta.kernel_backends.base import ExpertLayout

_MOE_RUNNER_BACKENDS = {
    "flashinfer_trtllm": ("trtllm", "none"),
    "flashinfer_trtllm_routed": ("trtllm", "none"),
    "flashinfer_mxfp4": ("trtllm", "none"),
    "flashinfer_cutedsl": ("cutedsl", "none"),
    "flashinfer_megamoe": ("megamoe", "flashinfer_megamoe"),
}


def identify_expert_layout(layer, runner_backend: str) -> ExpertLayout:
    if runner_backend not in _MOE_RUNNER_BACKENDS:
        raise ValueError(
            f"unsupported GPU delta MoE runner backend: {runner_backend}; supported: {sorted(_MOE_RUNNER_BACKENDS)}"
        )
    storage, _ = _MOE_RUNNER_BACKENDS[runner_backend]
    return flashinfer.identify_layout(layer, storage)


def admits_moe_dispatcher(runner_backend: str, a2a_backend: str) -> bool:
    return runner_backend in _MOE_RUNNER_BACKENDS and a2a_backend == _MOE_RUNNER_BACKENDS[runner_backend][1]
