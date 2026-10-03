import json
from collections import defaultdict

import yaml
from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.compact import ConfigSnapshotBases, compact_config_snapshot
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotCase
from miles.utils.test_utils.snapshot import snapshot_values


def dump_config_snapshot(case: ConfigSnapshotCase, *, bases: ConfigSnapshotBases | None = None) -> str:
    value = compact_config_snapshot(case=case, bases=bases) if bases is not None else case
    return _dump_snapshot(snapshot_values(value.model_dump(exclude_defaults=True) if bases is not None else value))


def _dump_snapshot(value: JsonValue) -> str:
    return yaml.dump(value, Dumper=_SnapshotDumper, sort_keys=True, allow_unicode=True, width=120)


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
        if len(encoded) <= 100 and "\\n" not in encoded:
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
