from pathlib import Path

import yaml

from miles.utils.test_utils.snapshot_yaml import dump_snapshot_delta, expand_snapshot_delta

_DELTA_HEADER = "# miles-snapshot-delta-v1\n"


def read_snapshot(path: Path) -> str:
    value = path.read_text()
    if not value.startswith(_DELTA_HEADER):
        return value
    base = _read_base(path=path, delta=value)
    return expand_snapshot_delta(base=base, delta=value)


def write_snapshot(*, path: Path, actual: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(actual)


def compact_snapshot_text(*, base: str, actual: str, base_name: str) -> str:
    delta = _DELTA_HEADER + dump_snapshot_delta(base=base, actual=actual, base_name=base_name)
    return delta if len(delta.encode()) < len(actual.encode()) else actual


def _read_base(*, path: Path, delta: str) -> str:
    base_path = Path(yaml.safe_load(delta)["base"])
    assert not base_path.is_absolute(), "Snapshot base must be a relative path"
    base = (path.parent / base_path).read_text()
    assert not base.startswith(_DELTA_HEADER), "Nested text snapshot bases are not supported"
    return base
