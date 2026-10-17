import difflib
import hashlib
import json
from collections import defaultdict

import yaml
from pydantic import JsonValue


def dump_snapshot_yaml(value: JsonValue) -> str:
    result = yaml.dump(value, Dumper=_SnapshotDumper, sort_keys=True, allow_unicode=True, width=120)
    assert_snapshot_equal(expected=value, actual=yaml.safe_load(result))
    return result


def dump_snapshot_delta(*, base: str, actual: str, base_name: str) -> str:
    before, after = base.splitlines(keepends=True), actual.splitlines(keepends=True)
    edits = [
        {"at": i, "delete": j - i, "insert": "".join(after[start:end])}
        for tag, i, j, start, end in difflib.SequenceMatcher(a=before, b=after).get_opcodes()
        if tag != "equal"
    ]
    result = dump_snapshot_yaml(
        {"base": base_name, "base_sha256": hashlib.sha256(base.encode()).hexdigest(), "edits": edits}
    )
    assert expand_snapshot_delta(base=base, delta=result) == actual, "Snapshot delta roundtrip failed"
    return result


def expand_snapshot_delta(*, base: str, delta: str) -> str:
    value = yaml.safe_load(delta)
    assert set(value) == {"base", "base_sha256", "edits"}
    assert value["base_sha256"] == hashlib.sha256(base.encode()).hexdigest(), "Snapshot base has changed"
    lines = base.splitlines(keepends=True)
    result: list[str] = []
    position = 0
    for edit in value["edits"]:
        assert set(edit) == {"at", "delete", "insert"}
        start, count = edit["at"], edit["delete"]
        assert type(start) is int and type(count) is int
        assert position <= start <= start + count <= len(lines)
        assert isinstance(edit["insert"], str)
        result.extend(lines[position:start])
        result.append(edit["insert"])
        position = start + count
    result.extend(lines[position:])
    return "".join(result)


def assert_snapshot_equal(*, expected: JsonValue, actual: JsonValue) -> None:
    assert json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True), "Snapshot roundtrip failed"


class _SnapshotDumper(yaml.SafeDumper):
    def represent(self, data: JsonValue) -> None:
        data = _share_values(value=data, cache={})
        self._shared_mappings = _find_shared_mappings(data)
        super().represent(data)

    def represent_dict(self, value: dict[str, JsonValue]) -> yaml.nodes.MappingNode:
        shared = self._shared_mappings.get(id(value), [])
        omitted = {key for group in shared for key in group}
        node = self.represent_mapping(
            "tag:yaml.org,2002:map", {key: item for key, item in value.items() if key not in omitted}
        )
        if shared:
            references = self.represent_data(shared[0] if len(shared) == 1 else shared)
            node.value.insert(0, (yaml.nodes.ScalarNode("tag:yaml.org,2002:merge", "<<"), references))
        encoded = json.dumps(value, ensure_ascii=False)
        scalar_values = all(not isinstance(item, (dict, list)) for item in value.values())
        if (len(encoded) <= 240 or scalar_values) and "\\n" not in encoded:
            node.flow_style = True
        return node

    def represent_list(self, value: list[JsonValue]) -> yaml.nodes.SequenceNode:
        scalar_items = all(not isinstance(item, (dict, list)) for item in value)
        compact = scalar_items and (len(value) <= 8 or all(not isinstance(item, str) for item in value))
        return self.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=compact)

    def represent_str(self, value: str) -> yaml.nodes.ScalarNode:
        return self.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


def _share_values(*, value: JsonValue, cache: dict[str, JsonValue]) -> JsonValue:
    if isinstance(value, dict):
        value = {key: _share_values(value=item, cache=cache) for key, item in sorted(value.items())}
    elif isinstance(value, list):
        value = [_share_values(value=item, cache=cache) for item in value]
    else:
        return value
    key = json.dumps(value, sort_keys=True)
    return cache.setdefault(key, value) if len(key) > 40 else value


def _find_shared_mappings(value: JsonValue) -> dict[int, list[dict[str, JsonValue]]]:
    mappings: dict[int, dict[str, JsonValue]] = {}
    _collect_mappings(value=value, mappings=mappings)
    uses: dict[tuple[str, str], list[int]] = defaultdict(list)
    entries: dict[tuple[str, str], JsonValue] = {}
    for identifier, mapping in mappings.items():
        for key, item in mapping.items():
            signature = (key, json.dumps(item, sort_keys=True))
            uses[signature].append(identifier)
            entries[signature] = item

    groups: dict[tuple[int, ...], dict[str, JsonValue]] = defaultdict(dict)
    for signature, identifiers in uses.items():
        if len(identifiers) > 1:
            groups[tuple(identifiers)][signature[0]] = entries[signature]

    result: dict[int, list[dict[str, JsonValue]]] = defaultdict(list)
    for identifiers, mapping in groups.items():
        if len(mapping) < 2 or (len(json.dumps(mapping)) - 12) * (len(identifiers) - 1) < 40:
            continue
        for identifier in identifiers:
            result[identifier].append(mapping)
    return dict(result)


def _collect_mappings(*, value: JsonValue, mappings: dict[int, dict[str, JsonValue]]) -> None:
    if isinstance(value, dict):
        if id(value) in mappings:
            return
        mappings[id(value)] = value
        for item in value.values():
            _collect_mappings(value=item, mappings=mappings)
    elif isinstance(value, list):
        for item in value:
            _collect_mappings(value=item, mappings=mappings)


_SnapshotDumper.add_representer(dict, _SnapshotDumper.represent_dict)
_SnapshotDumper.add_representer(list, _SnapshotDumper.represent_list)
_SnapshotDumper.add_representer(str, _SnapshotDumper.represent_str)
