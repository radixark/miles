"""Generic bridge between frozen dataclasses and argparse.

``DataclassArgparseBridge`` eliminates the need for each dataclass to
hand-write ``register_argparse`` / ``from_namespace`` / ``to_cli_args``
methods.  Only the dataclass field definitions are needed — the bridge
derives everything else from ``get_type_hints()``.

Supported field types: ``str``, ``int``, ``float``, ``bool``,
``str | None``, ``int | None``, ``float | None``.
"""

from __future__ import annotations

import argparse
import dataclasses
import types
from typing import Generic, TypeVar, get_type_hints

T = TypeVar("T")

_SCALAR_TYPES: dict[type, type] = {str: str, int: int, float: float}


def _is_bool(tp: type) -> bool:
    return tp is bool


def _is_optional(tp: type) -> tuple[bool, type | None]:
    """Return ``(True, inner)`` if *tp* is ``X | None``, else ``(False, None)``."""
    if not isinstance(tp, types.UnionType):
        return False, None

    args: tuple[type, ...] = tp.__args__
    non_none: list[type] = [a for a in args if a is not type(None)]
    if type(None) not in args or len(non_none) != 1:
        return False, None

    return True, non_none[0]


class DataclassArgparseBridge(Generic[T]):
    """Bi-directional converter between a frozen dataclass and argparse.

    Parameters
    ----------
    dataclass_type:
        The dataclass class itself (not an instance).
    prefix:
        CLI flag prefix.  ``"script"`` → ``--script-field-name``;
        ``""`` → ``--field-name``.
    group_title:
        Title used for the argparse argument group.
    """

    def __init__(
        self,
        dataclass_type: type[T],
        *,
        prefix: str,
        group_title: str | None = None,
    ) -> None:
        if not dataclasses.is_dataclass(dataclass_type):
            raise TypeError(f"{dataclass_type!r} is not a dataclass")

        self._cls: type[T] = dataclass_type
        self._prefix: str = prefix
        self._group_title: str = group_title or f"{dataclass_type.__name__} args"
        self._hints: dict[str, type] = get_type_hints(dataclass_type)

    def _flag(self, field_name: str) -> str:
        stem: str = field_name.replace("_", "-")
        if self._prefix:
            return f"--{self._prefix}-{stem}"
        return f"--{stem}"

    def _dest(self, field_name: str) -> str:
        if self._prefix:
            return f"{self._prefix}_{field_name}"
        return field_name

    def register_on_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments derived from the dataclass fields to *parser*."""
        group: argparse._ArgumentGroup = parser.add_argument_group(self._group_title)

        for field in dataclasses.fields(self._cls):
            flag: str = self._flag(field.name)
            dest: str = self._dest(field.name)
            tp: type = self._hints[field.name]

            if _is_bool(tp):
                group.add_argument(flag, dest=dest, action="store_true", default=False)
                continue

            is_opt, inner = _is_optional(tp)
            if is_opt:
                if inner not in _SCALAR_TYPES:
                    raise TypeError(
                        f"Unsupported optional inner type {inner!r} for field {field.name}"
                    )
                group.add_argument(flag, dest=dest, type=_SCALAR_TYPES[inner], default=None)
                continue

            if tp in _SCALAR_TYPES:
                has_default: bool = field.default is not dataclasses.MISSING
                kwargs: dict[str, object] = {"dest": dest, "type": _SCALAR_TYPES[tp]}
                if has_default:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
                group.add_argument(flag, **kwargs)  # type: ignore[arg-type]
                continue

            raise TypeError(f"Unsupported field type {tp!r} for field {field.name}")

    def from_namespace(self, namespace: argparse.Namespace) -> T:
        """Construct a dataclass instance from a parsed ``Namespace``."""
        kwargs: dict[str, object] = {}
        for field in dataclasses.fields(self._cls):
            kwargs[field.name] = getattr(namespace, self._dest(field.name))
        return self._cls(**kwargs)  # type: ignore[call-arg]

    def to_cli_args(self, instance: T) -> str:
        """Serialize a dataclass instance into a CLI argument string."""
        parts: list[str] = []

        for field in dataclasses.fields(self._cls):  # type: ignore[arg-type]
            value: object = getattr(instance, field.name)
            flag: str = self._flag(field.name)
            tp: type = self._hints[field.name]

            if _is_bool(tp):
                if value:
                    parts.append(flag)
            elif value is not None:
                parts.append(f"{flag} {value}")

        return " ".join(parts)
