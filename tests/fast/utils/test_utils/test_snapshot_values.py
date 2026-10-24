import argparse
import functools
import re
from datetime import date
from pathlib import Path
from signal import Signals
from typing import Any

import pytest

from miles.utils.test_utils.snapshot import snapshot_values


class TestSnapshotValues:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ((1, "x"), {"$tuple": [1, "x"]}),
            ({3, 1, 2}, {"$set": [1, 2, 3]}),
            (frozenset(), {"$set": []}),
            (Path("relative/file"), {"$path": "relative/file"}),
            (date(2026, 10, 17), {"$date": "2026-10-17"}),
            (range(1, 10, 2), {"$range": [1, 10, 2]}),
            (Signals.SIGTERM, {"$enum": "signal.Signals", "name": "SIGTERM"}),
            (re.compile("pattern", re.I), {"$regex": "pattern", "flags": re.I | re.U}),
        ],
    )
    def test_tagged_values_remain_distinguishable_from_plain_json_values(self, value: Any, expected: Any) -> None:
        """Snapshot normalization retains semantic type information that JSON alone would lose."""
        assert snapshot_values(value) == expected

    def test_partial_callables_preserve_bound_arguments(self) -> None:
        """A changed callback binding remains visible even when its function is unchanged."""
        result = snapshot_values(functools.partial(pow, 2, exp=3))

        assert result == {
            "$partial": {"$callable": "builtins.pow"},
            "args": {"$tuple": [2]},
            "keywords": {"exp": 3},
        }

    def test_namespace_nesting_is_normalized_recursively(self) -> None:
        """Nested namespaces retain typed leaves instead of unstable object representations."""
        value = argparse.Namespace(child=argparse.Namespace(paths=(Path("a"), Path("b"))))

        assert snapshot_values(value) == {"child": {"paths": {"$tuple": [{"$path": "a"}, {"$path": "b"}]}}}

    @pytest.mark.parametrize("value", [{1: "integer key"}, {"nested": {False: "boolean key"}}, object()])
    def test_unsupported_values_fail_instead_of_becoming_lossy_strings(self, value: Any) -> None:
        """Unsupported objects and non-string keys must not produce misleading stable snapshots."""
        with pytest.raises(TypeError):
            snapshot_values(value)
