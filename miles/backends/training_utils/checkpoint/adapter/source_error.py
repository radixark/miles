from collections.abc import Iterable
from pathlib import Path


class AdapterSourceError(ValueError):
    """A source the caller named is not one this run can load: unrecognized, malformed, or another parametrization.

    Raised by the library's source checks only, so a caller can tell a refused source from a storage fault or a bug.
    """


def require_fields(fields: dict, names: Iterable[str], path: Path) -> None:
    """A source file without a field it must have is the source's fault, not a KeyError in the reader."""
    missing = [name for name in names if name not in fields]
    if missing:
        raise AdapterSourceError(f"{path} lacks {missing}")
