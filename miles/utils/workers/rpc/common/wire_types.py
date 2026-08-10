from __future__ import annotations

from argparse import Namespace
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer


def _as_namespace(value: Any) -> Namespace:
    if isinstance(value, Namespace):
        return value
    assert isinstance(value, dict), f"a namespace argument arrives as a mapping, got {type(value).__name__}"
    return Namespace(**value)


def _as_mapping(value: Namespace) -> dict[str, Any]:
    return vars(value)


WireNamespace = Annotated[
    Any,
    BeforeValidator(_as_namespace),
    PlainSerializer(_as_mapping, return_type=dict[str, Any]),
]
