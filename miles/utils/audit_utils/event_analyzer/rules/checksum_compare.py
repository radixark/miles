from collections.abc import Iterable
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ChecksumMismatchIssue(FrozenStrictBaseModel):
    key: str
    label_a: str
    label_b: str
    value_a: str
    value_b: str


def compare_flat_dicts(
    a: dict[str, Any],
    b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> Iterable[ChecksumMismatchIssue]:
    """Compare two flat dicts and yield one mismatch per differing key."""
    all_keys = sorted(set(a.keys()) | set(b.keys()))

    for key in all_keys:
        value_a = a.get(key, "<missing>")
        value_b = b.get(key, "<missing>")
        if value_a != value_b:
            yield ChecksumMismatchIssue(
                key=key,
                label_a=label_a,
                label_b=label_b,
                value_a=str(value_a),
                value_b=str(value_b),
            )


def flatten_nested(obj: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict/list into a flat dict with dot-separated keys. Keeps all primitive leaf values."""
    result: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            result.update(flatten_nested(v, prefix=child_prefix))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            result.update(flatten_nested(v, prefix=f"{prefix}[{i}]"))
    else:
        result[prefix] = obj

    return result
