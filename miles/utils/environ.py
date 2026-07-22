import os

_printed_experimental_rollout_refactor = False


def enable_experimental_rollout_refactor() -> bool:
    result = bool(int(os.environ.get("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "0")))

    global _printed_experimental_rollout_refactor
    if result and not _printed_experimental_rollout_refactor:
        print("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 is enabled (experimental feature)")
        _printed_experimental_rollout_refactor = True

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


_printed_experimental_ft_trainer = False


def enable_experimental_ft_trainer() -> bool:
    raw = os.environ.get("MILES_EXPERIMENTAL_FT_TRAINER", "0").lower()
    result = raw in ("1", "true", "on", "yes")

    global _printed_experimental_ft_trainer
    if result and not _printed_experimental_ft_trainer:
        print("MILES_EXPERIMENTAL_FT_TRAINER=1 is enabled (experimental feature)")
        _printed_experimental_ft_trainer = True

    return result
