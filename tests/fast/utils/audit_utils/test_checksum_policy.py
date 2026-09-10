import pytest

from miles.utils.audit_utils.checksum_policy import checksum_movement_skip_reasons


class TestChecksumMovementPolicy:
    @pytest.mark.parametrize(
        "lora,interval,expected",
        [
            (False, 1, []),
            (True, 1, ["lora_base_weights"]),
            (False, 2, ["update_interval"]),
            (True, 2, ["lora_base_weights", "update_interval"]),
        ],
    )
    def test_unsupported_modes_record_every_skip_reason(self, lora: bool, interval: int, expected: list[str]) -> None:
        """Only full-weight updates at every step require adjacent-version movement."""
        assert checksum_movement_skip_reasons(lora_enabled=lora, update_weights_interval=interval) == expected

    def test_invalid_interval_is_rejected(self) -> None:
        """An invalid interval must not silently become an excluded configuration."""
        with pytest.raises(ValueError, match="positive"):
            checksum_movement_skip_reasons(lora_enabled=False, update_weights_interval=0)
