from __future__ import annotations

import hashlib

CHART_NAME = "miles-run"

RELEASE_DIGEST_LENGTH = 6


def release_prefix(release: str, *, chart_name: str, budget: int) -> str:
    name = release if chart_name in release else f"{release}-{chart_name}"
    if len(name) <= budget:
        return name
    digest = hashlib.blake2b(release.encode(), digest_size=RELEASE_DIGEST_LENGTH).hexdigest()
    return f"{_trim_suffix(name[: budget - (len(digest) + 1)], '-')}-{digest}"


def _trim_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value
