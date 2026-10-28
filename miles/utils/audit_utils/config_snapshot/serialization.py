import yaml

from miles.utils.audit_utils.config_snapshot.compact_codec import (
    CompactCode,
    CompactConfigSnapshotCase,
    ConfigSnapshotBases,
)
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotCase
from miles.utils.test_utils.snapshot import snapshot_values
from miles.utils.test_utils.snapshot_yaml import assert_snapshot_equal
from miles.utils.test_utils.snapshot_yaml import dump_snapshot_yaml as _dump_snapshot


def dump_config_snapshot(case: ConfigSnapshotCase, *, bases: ConfigSnapshotBases | None = None) -> str:
    value = CompactCode.encode(case=case, bases=bases) if bases is not None else case
    result = _dump_snapshot(snapshot_values(value.model_dump(exclude_defaults=True) if bases is not None else value))
    parsed = yaml.safe_load(result)
    restored = (
        CompactCode.decode(case=CompactConfigSnapshotCase.model_validate(parsed), bases=bases)
        if bases is not None
        else ConfigSnapshotCase.model_validate(parsed)
    )
    assert_snapshot_equal(expected=case.model_dump(), actual=restored.model_dump())
    return result
