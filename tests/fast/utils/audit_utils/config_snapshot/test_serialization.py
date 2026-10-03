import json

import pytest
import yaml

from miles.utils.audit_utils.config_snapshot.compact import (
    CompactConfigSnapshotCase,
    ConfigSnapshotBases,
    expand_config_snapshot,
)
from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotCase,
    ConfigSnapshotDelta,
    ConfigSnapshotProcess,
)
from miles.utils.audit_utils.config_snapshot.serialization import _dump_snapshot, dump_config_snapshot
from miles.utils.test_utils.snapshot import dump_snapshot, snapshot_values


@pytest.mark.parametrize("suffix", ["", "\n", "\n\n"])
def test_multiline_values_remain_readable_and_round_trip_exactly(suffix: str) -> None:
    """Literal blocks preserve multiline configuration values without inserting blank display lines."""
    value = "first\nsecond: 新" + suffix
    case = ConfigSnapshotCase(
        processes={
            "trainer": ConfigSnapshotProcess(
                ranks=[0],
                base={"value": "old"},
                diffs={"stage": ConfigSnapshotDelta(set={"/value": value}, remove=[])},
            )
        }
    )
    ordinary = dump_snapshot(case)

    actual = dump_config_snapshot(case)

    assert "/value: |" in actual
    assert "first\n            second: 新" in actual
    assert yaml.safe_load(actual) == snapshot_values(case)
    assert dump_snapshot(case) == ordinary


@pytest.mark.parametrize("distinct", [None, False, 0, 0.0, [], {}])
def test_shared_mapping_values_preserve_types_and_missing_fields(distinct: object) -> None:
    """Sharing mapping entries preserves missing fields, literal merge keys, and distinct JSON types."""
    common = {"host": "$HOST", "port": "$PORT", "enabled": False, "nullable": None}
    value = {
        "first": {**common, "<<": "literal key", "value": distinct},
        "second": {**common, "<<": "literal key", "value": True},
        "missing": common,
        "repeated": [{**common, "index": index} for index in range(4)],
    }
    expected = json.dumps(value, sort_keys=True)

    actual = _dump_snapshot(value)

    assert "&id" in actual
    assert json.dumps(yaml.safe_load(actual), sort_keys=True) == expected
    assert json.dumps(value, sort_keys=True) == expected


def test_shared_mapping_output_is_stable_across_input_order() -> None:
    """Mapping insertion order does not change shared YAML or scalar list ordering."""
    common = {"host": "$HOST", "port": "$PORT", "enabled": False, "nullable": None}
    value = {"first": {**common, "ranks": [3, 1, 2]}, "second": {**common, "ranks": [0, 4]}}
    reordered = {name: dict(reversed(item.items())) for name, item in reversed(value.items())}

    actual = _dump_snapshot(value)

    assert actual == _dump_snapshot(reordered)
    assert actual == _dump_snapshot(yaml.safe_load(actual))
    assert yaml.safe_load(actual) == value


def test_compact_defaults_keep_empty_payloads_and_unchanged_stages() -> None:
    """Omitted envelope defaults restore empty ranks and stages without dropping explicit payload values."""
    bases = ConfigSnapshotBases(templates={"default": {}})
    case = ConfigSnapshotCase(
        processes={
            "worker": ConfigSnapshotProcess(
                ranks=[],
                base={"empty": [], "mapping": {}, "none": None, "false": False},
                diffs={"unchanged": ConfigSnapshotDelta(set={}, remove=[])},
            )
        }
    )

    actual = dump_config_snapshot(case, bases=bases)
    parsed = CompactConfigSnapshotCase.model_validate(yaml.safe_load(actual))
    restored = expand_config_snapshot(case=parsed, bases=bases)

    assert "remove:" not in actual
    assert "ranks:" not in actual
    assert json.dumps(restored.model_dump(), sort_keys=True) == json.dumps(case.model_dump(), sort_keys=True)


def test_short_mapping_records_remain_readable_and_round_trip_exactly() -> None:
    """Short records use flow YAML while multiline values remain literal blocks."""
    value = {"short": {"rank": 0, "enabled": False, "value": None}, "text": {"value": "first\nsecond"}}

    actual = _dump_snapshot(value)

    assert "short: {enabled: false, rank: 0, value: null}" in actual
    assert "value: |" in actual
    assert json.dumps(yaml.safe_load(actual), sort_keys=True) == json.dumps(value, sort_keys=True)
    assert actual == _dump_snapshot(yaml.safe_load(actual))
