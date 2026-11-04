import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.compact_codec import DeltaCodec
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotDelta


class TestDeltaDecodeBoundaries:
    @pytest.mark.parametrize(
        "base,path,message",
        [
            ({}, "invalid", "Expected a non-root JSON Pointer"),
            ({"value": 1}, "/value/child/leaf", "traverses a non-mapping"),
            ({"value": 1}, "/value/child", "parent is not a mapping"),
        ],
    )
    def test_invalid_override_paths_fail_before_modifying_the_base(
        self, base: JsonValue, path: str, message: str
    ) -> None:
        """Invalid structural paths cannot silently reinterpret a compact snapshot override."""
        overrides = ConfigSnapshotDelta(set={path: 3}, remove=[])
        before = dict(base)

        with pytest.raises(ValueError, match=message):
            DeltaCodec.decode(base=base, overrides=overrides)

        assert base == before

    def test_mutating_decoded_values_cannot_change_either_the_template_or_override_payload(self) -> None:
        """Expanded snapshots own their nested data independently of reusable input templates and overrides."""
        base = {"nested": {"values": [1]}}
        overrides = ConfigSnapshotDelta(set={"/added": {"values": [2]}}, remove=[])

        result = DeltaCodec.decode(base=base, overrides=overrides)
        result["nested"]["values"].append(9)
        result["added"]["values"].append(8)

        assert base == {"nested": {"values": [1]}}
        assert overrides.set == {"/added": {"values": [2]}}
