import json

import pytest
import yaml
from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.compact import (
    CompactConfigSnapshotCase,
    ConfigSnapshotBases,
    apply_snapshot_delta,
    compact_config_snapshot,
    expand_config_snapshot,
    make_snapshot_delta,
)
from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotCase,
    ConfigSnapshotDelta,
    ConfigSnapshotProcess,
)
from miles.utils.audit_utils.config_snapshot.serialization import dump_config_snapshot


@pytest.mark.parametrize(
    "base,actual",
    [
        ({"missing": None, "nullable": 1}, {"nullable": None, "added": None}),
        ({"nested": {"old": 1}, "list": [1]}, {"nested": {}, "list": []}),
        ({"value": 1, "list": [1, 0]}, {"value": True, "list": [True, False]}),
        ({"value": 1}, {"value": 1.0}),
        ({"a/b": {"~key": 1, "": 2}}, {"a/b": {"~key": None, "new": 3}}),
        ({"value": None}, {"value": {"nested": []}}),
        ({"value": {"nested": []}}, {"value": None}),
        ({"value": 1}, None),
        (None, {"value": 1}),
        ([1, 2], [3]),
    ],
)
def test_expansion_preserves_every_value_and_type(base: JsonValue, actual: JsonValue) -> None:
    """Template overrides preserve missing keys, nulls, empty containers, escaped keys, and scalar types."""
    bases = ConfigSnapshotBases(templates={"default": base})
    before = json.dumps(bases.model_dump(), sort_keys=True)
    case = ConfigSnapshotCase(processes={"trainer": ConfigSnapshotProcess(ranks=[0], base=actual, diffs={})})

    encoded = dump_config_snapshot(case, bases=bases)
    expanded = expand_config_snapshot(
        case=CompactConfigSnapshotCase.model_validate(yaml.safe_load(encoded)), bases=bases
    )

    assert json.dumps(expanded.model_dump(), sort_keys=True) == json.dumps(case.model_dump(), sort_keys=True)
    assert json.dumps(bases.model_dump(), sort_keys=True) == before


def test_removal_is_distinct_from_setting_null() -> None:
    """Absent keys are removed explicitly while null remains an ordinary assigned value."""
    bases = ConfigSnapshotBases(templates={"default": {"deleted": None, "nullable": 1}})
    case = ConfigSnapshotCase(
        processes={"trainer": ConfigSnapshotProcess(ranks=[0], base={"nullable": None}, diffs={})}
    )

    process = compact_config_snapshot(case=case, bases=bases).processes["trainer"]

    assert process.model_dump()["overrides"] == {"set": {"/nullable": None}, "remove": ["/deleted"]}


def test_template_choice_is_smallest_and_independent_of_insertion_order() -> None:
    """The nearest template wins and template names resolve equal-size ties deterministically."""
    case = ConfigSnapshotCase(processes={"trainer": ConfigSnapshotProcess(ranks=[0], base={"value": 2}, diffs={})})
    templates = {"far": {"different": 3}, "z-near": {"value": 2}, "a-near": {"value": 2}}

    forward = compact_config_snapshot(case=case, bases=ConfigSnapshotBases(templates=templates))
    backward = compact_config_snapshot(
        case=case, bases=ConfigSnapshotBases(templates=dict(reversed(templates.items())))
    )

    assert forward.model_dump() == backward.model_dump()
    assert forward.processes["trainer"].model_dump()["template"] == "a-near"


def test_shared_case_overrides_preserve_each_process_and_stage() -> None:
    """Common nested values occur once while process-specific changes and every stage survive."""
    common = {f"key_{index}": index for index in range(20)}
    delta = ConfigSnapshotDelta(set={"/value": None}, remove=["/deleted"])
    case = ConfigSnapshotCase(
        processes={
            str(rank): ConfigSnapshotProcess(
                ranks=[rank],
                base={"common": common, "value": rank, "deleted": None},
                diffs={"stage": delta, "unchanged": ConfigSnapshotDelta(set={}, remove=[])},
            )
            for rank in range(3)
        }
    )
    bases = ConfigSnapshotBases(templates={"default": {"value": -1}})

    compact = compact_config_snapshot(case=case, bases=bases)

    assert compact.shared_overrides["default"].set == {"/common": common, "/deleted": None}
    assert all(process.overrides.set == {"/value": int(name)} for name, process in compact.processes.items())
    assert all(
        {stage: change.model_dump() for stage, change in process.diffs.items()}
        == {"stage": delta.model_dump(), "unchanged": {"set": {}, "remove": []}}
        for process in compact.processes.values()
    )
    assert expand_config_snapshot(case=compact, bases=bases).model_dump() == case.model_dump()


def test_shared_overrides_do_not_equate_boolean_integer_and_float() -> None:
    """Common-field extraction does not merge equal-looking values of different JSON types."""
    common = {f"key_{index}": index for index in range(20)}
    case = ConfigSnapshotCase(
        processes={
            str(index): ConfigSnapshotProcess(ranks=[index], base={"common": common, "value": value}, diffs={})
            for index, value in enumerate([True, 1, 1.0])
        }
    )
    bases = ConfigSnapshotBases(templates={"default": {}})

    compact = compact_config_snapshot(case=case, bases=bases)
    expanded = expand_config_snapshot(case=compact, bases=bases)

    assert "/value" not in compact.shared_overrides["default"].set
    assert json.dumps(expanded.model_dump(), sort_keys=True) == json.dumps(case.model_dump(), sort_keys=True)


def test_unprofitable_shared_overrides_remain_local() -> None:
    """A tiny duplicate field does not add a larger common-override section."""
    case = ConfigSnapshotCase(
        processes={str(index): ConfigSnapshotProcess(ranks=[index], base={"x": 1}, diffs={}) for index in range(2)}
    )
    bases = ConfigSnapshotBases(templates={"default": {}})

    compact = compact_config_snapshot(case=case, bases=bases)

    assert compact.shared_overrides == {}
    assert expand_config_snapshot(case=compact, bases=bases).model_dump() == case.model_dump()


@pytest.mark.parametrize(
    "base,actual",
    [
        ({"a": 1, "b": None}, {"a": None, "new": []}),
        ({"nested": {"a/b": [1]}}, {"nested": {"a/b": [True]}}),
        ({"x": 1}, {"x": 1.0}),
        ({"x": {}}, {"x": []}),
        (None, {"x": None}),
        ({"x": 1}, None),
    ],
)
def test_structured_stage_delta_preserves_types_and_removals(base: JsonValue, actual: JsonValue) -> None:
    """Structural stage changes preserve nulls, missing fields, escaped paths and type transitions."""
    delta = make_snapshot_delta(base=base, actual=actual)
    restored = apply_snapshot_delta(base=base, overrides=delta)

    assert json.dumps(restored, sort_keys=True) == json.dumps(actual, sort_keys=True)


def test_empty_template_collection_is_rejected() -> None:
    """Missing shared templates cannot silently fall back to uncompressed output."""
    with pytest.raises(ValueError, match="must not be empty"):
        compact_config_snapshot(case=ConfigSnapshotCase(processes={}), bases=ConfigSnapshotBases(templates={}))
