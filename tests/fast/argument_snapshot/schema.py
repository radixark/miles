import argparse
import enum
import functools
import inspect
import re
from pathlib import Path
from typing import Any

import yaml


def snapshot_parser(parser: argparse.ArgumentParser) -> dict[str, Any]:
    destinations = dict.fromkeys([action.dest for action in parser._actions] + list(parser._defaults))
    return {
        "parser": parser,
        "effective_defaults": {dest: parser.get_default(dest) for dest in destinations},
    }


def dump_snapshot(value: Any) -> str:
    return yaml.dump(value, Dumper=_SnapshotDumper, sort_keys=True, allow_unicode=True, width=120)


def _encode(value: Any) -> dict[str, Any]:
    if isinstance(value, (argparse._ActionsContainer, argparse.Action)):
        return {
            "$class": _qualified_name(type(value)),
            "state": vars(value),
        }
    if isinstance(value, enum.Enum):
        return {"$enum": _qualified_name(type(value)), "name": value.name}
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, tuple):
        return {"$tuple": list(value)}
    if isinstance(value, range):
        return {"$range": [value.start, value.stop, value.step]}
    if isinstance(value, re.Pattern):
        return {"$regex": value.pattern, "flags": value.flags}
    if isinstance(value, functools.partial):
        return {"$partial": value.func, "args": value.args, "keywords": value.keywords}
    if isinstance(value, type) or inspect.isfunction(value) or inspect.isbuiltin(value):
        return {"$callable": _qualified_name(value)}
    raise TypeError(f"Unsupported snapshot value type: {_qualified_name(type(value))}")


def _qualified_name(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


class _SnapshotDumper(yaml.SafeDumper):
    def represent_special(self, value: Any) -> yaml.MappingNode:
        return self.represent_mapping("tag:yaml.org,2002:map", _encode(value))


_SnapshotDumper.add_representer(tuple, _SnapshotDumper.represent_special)
_SnapshotDumper.add_multi_representer(object, _SnapshotDumper.represent_special)
