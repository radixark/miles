from typing import Literal

ChecksumMovementSkipReason = Literal["lora_base_weights", "update_interval"]


def checksum_movement_skip_reasons(
    *, lora_enabled: bool, update_weights_interval: int
) -> list[ChecksumMovementSkipReason]:
    if update_weights_interval < 1:
        raise ValueError("Weight update interval must be positive")
    reasons: list[ChecksumMovementSkipReason] = []
    if lora_enabled:
        reasons.append("lora_base_weights")
    if update_weights_interval != 1:
        reasons.append("update_interval")
    return reasons
