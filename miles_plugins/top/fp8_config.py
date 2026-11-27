"""Read checkpoint quantization and validate the FP8 format shared with serving."""


def checkpoint_is_block_fp8(hf_checkpoint: str) -> bool:
    """Read the checkpoint's quantization; reject formats TOP cannot pair."""
    from sglang.srt.true_on_policy.fp8 import checkpoint_block

    from miles.utils.hf_utils.config import load_hf_config

    quantization = getattr(
        load_hf_config(hf_checkpoint), "quantization_config", None
    )  # config-access-exempt: only quantized HF checkpoints carry quantization_config
    if quantization is None:
        return False
    if checkpoint_block(quantization) is None:
        raise NotImplementedError(
            "[top] only unquantized or block-wise FP8 (E4M3, dynamic, 128x128, or DeepSeek-V4.1's 32x32 UE8M0) "
            "checkpoints are supported"
        )
    return True
