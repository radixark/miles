import os

_printed_legacy_rollout_v1 = False


def use_legacy_rollout_v1() -> bool:
    result = bool(int(os.environ.get("MILES_USE_LEGACY_ROLLOUT_V1", "0")))

    global _printed_legacy_rollout_v1
    if result and not _printed_legacy_rollout_v1:
        print("MILES_USE_LEGACY_ROLLOUT_V1=1 is enabled: using the deprecated v1 rollout path")
        _printed_legacy_rollout_v1 = True

    return result


def default_fp8_block_scaling_fp32_scales() -> str:
    """Default for NVTE_FP8_BLOCK_SCALING_FP32_SCALES, decided by hardware.

    On Blackwell (SM100+), TE emulates the blockwise FP8 recipe with MXFP8,
    which requires power-of-two scales, so FP32 scales must stay disabled.
    """
    import torch

    if not torch.cuda.is_available():
        return "1"
    major, _minor = torch.cuda.get_device_capability()
    return "0" if major >= 10 else "1"


def default_train_inductor_autotune_env() -> dict[str, str]:
    """Inductor autotune env for training actors, decided by platform.

    The ROCm image inherits TORCHINDUCTOR_MAX_AUTOTUNE=1 from the sglang base image.
    With it on, compiled reductions benchmark many launch configs and the two forward
    passes of one step can pick different ones, so log-probs stop being bitwise identical.
    """
    if os.environ.get("MILES_HARDWARE_PLATFORM") != "rocm":
        return {}
    return {
        "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
        "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "0",
    }
