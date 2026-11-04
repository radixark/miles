import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pathlib import Path

import torch
from tests.fast.utils.test_utils.snapshot_value_fixtures import _SnapshotDataclass, _SnapshotModel

from miles.utils.test_utils.snapshot import snapshot_values


class TestSnapshotTypeBoundaries:
    @pytest.mark.parametrize(
        "value", [_SnapshotModel(location=Path("actual")), _SnapshotDataclass(location=Path("actual"))]
    )
    def test_nested_model_and_dataclass_values_retain_their_actual_typed_state(self, value: object) -> None:
        """Snapshotting reads typed object fields instead of a potentially lossy custom serialization."""
        assert snapshot_values({"container": [value]}) == {"container": [{"location": {"$path": "actual"}}]}

    @pytest.mark.parametrize("value,expected", [(torch.float32, "torch.float32"), (torch.device("cpu"), "cpu")])
    def test_torch_dtype_and_device_remain_distinguishable_from_plain_strings(
        self, value: object, expected: str
    ) -> None:
        """Backend dtype and device values retain an explicit type tag in snapshots."""
        assert snapshot_values(value) == {"$torch": expected}
        assert snapshot_values(expected) == expected
