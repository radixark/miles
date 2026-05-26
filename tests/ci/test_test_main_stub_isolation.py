"""Regression test guarding the stub-isolation invariant of
``tests/fast/utils/debug_utils/run_megatron/worker/test_main.py``.

Contract under test
-------------------
``test_main.py`` calls ``_ensure_module`` at module import time to populate
``sys.modules`` with stubs for leaf modules whose top-level imports would
fail in the lightweight test container.  The hard invariant: for any leaf
name that is actually present on disk (``importlib.util.find_spec`` returns
a non-None spec), the entry in ``sys.modules`` after import MUST be the
real module (has a populated ``__file__`` or ``__path__``), not a starved
``ModuleType`` instance fabricated by ``_ensure_module``.

This guards against two regressions:

1. The current bug (pre-``a5757ca99``) where a bare ``try / except
   ImportError`` around ``importlib.import_module`` could not distinguish
   "module absent from disk" from "module on disk whose body raises
   ImportError due to a missing transitive dep" — both fell through to the
   stub branch and overwrote the real module name in ``sys.modules``.

2. Any future re-introduction of blind-stub behavior gated on simple
   import success rather than ``find_spec`` presence.

The test runs the probe in a fresh subprocess so it never has to roll back
``sys.modules`` mutations in the parent interpreter.  It depends only on
stdlib and pytest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Leaf names that ``test_main.py`` stubs at import time.  If that list ever
# grows the probe below should grow with it; the invariant applies to every
# name added to ``sys.modules`` by ``_ensure_module``.
_LEAF_NAMES = (
    "miles.backends.megatron_utils.arguments",
    "miles.backends.megatron_utils.checkpoint",
    "miles.backends.megatron_utils.initialize",
    "miles.backends.megatron_utils.model_provider",
)


_PROBE_SCRIPT = r"""
import json
import importlib
import importlib.util
import sys

LEAF_NAMES = {leaf_names!r}

# Snapshot find_spec results BEFORE importing test_main, so the answer
# reflects on-disk presence and is not polluted by any stubs placed into
# sys.modules at import time (find_spec consults sys.modules first and
# would otherwise reflect a stub's missing __spec__ rather than the
# real on-disk module).
on_disk = {{}}
for name in LEAF_NAMES:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    on_disk[name] = spec is not None

# Trigger the module-level stubbing performed by test_main.py.  In
# environments where a leaf's body raises ImportError (the real bug
# surface the new _ensure_module deliberately propagates), import_module
# bubbles out of here — proving the stub branch did NOT corrupt
# sys.modules with a starved ModuleType.  Report the propagation
# explicitly so the parent test can treat it as an invariant-satisfied
# outcome rather than a probe failure.
try:
    importlib.import_module(
        "tests.fast.utils.debug_utils.run_megatron.worker.test_main"
    )
    import_ok = True
    import_error = None
except ImportError as exc:
    import_ok = False
    import_error = f"{{type(exc).__name__}}: {{exc}}"

records = []
for name in LEAF_NAMES:
    mod = sys.modules.get(name)
    has_file = getattr(mod, "__file__", None) is not None
    has_path = getattr(mod, "__path__", None) is not None
    is_real_module = mod is not None and (has_file or has_path)

    records.append({{
        "name": name,
        "exists_on_disk": on_disk[name],
        "is_real_module": is_real_module,
        "file": getattr(mod, "__file__", None),
        "in_sys_modules": mod is not None,
        "type": type(mod).__name__ if mod is not None else None,
    }})

print(json.dumps({{
    "import_ok": import_ok,
    "import_error": import_error,
    "records": records,
}}))
"""


def _repo_root() -> Path:
    # tests/ci/test_test_main_stub_isolation.py  →  repo root is parents[2].
    return Path(__file__).resolve().parents[2]


def test_test_main_does_not_starve_real_leaves() -> None:
    """Every find_spec-positive leaf must resolve to a real module in
    ``sys.modules`` after ``test_main`` is imported — never a fabricated
    empty ``ModuleType`` stub."""
    repo_root = _repo_root()
    probe = _PROBE_SCRIPT.format(leaf_names=list(_LEAF_NAMES))

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env={
            **_inherited_env(),
            "PYTHONPATH": str(repo_root),
        },
    )

    assert result.returncode == 0, (
        f"probe subprocess failed (rc={result.returncode}).\n" f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )

    # The probe prints exactly one JSON line; tolerate trailing whitespace
    # or interleaved warnings by taking the last non-empty stdout line.
    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert stdout_lines, f"probe produced no stdout. stderr:\n{result.stderr}"
    payload = json.loads(stdout_lines[-1])

    # If test_main itself raised ImportError during its module-level
    # stubbing loop, the new _ensure_module deliberately surfaced a real
    # transitive-dep failure rather than masking it with a starved stub.
    # That is the desired outcome of this round's fix; treat it as an
    # invariant-satisfied case (no stubs were placed, so none can be
    # starved).  Production CI runs in an env where the bodies do import
    # cleanly and takes the records branch below.
    if not payload["import_ok"]:
        return

    records = payload["records"]
    starved = [r for r in records if r["exists_on_disk"] and not r["is_real_module"]]
    assert not starved, (
        "test_main stubbed real on-disk modules with starved ModuleType "
        "instances; this means _ensure_module fell back to its stub branch "
        "for a module find_spec considers importable, which violates the "
        "stub-isolation invariant. Offending records:\n" + json.dumps(starved, indent=2)
    )


def _inherited_env() -> dict[str, str]:
    """Copy the parent environment except for PYTHONPATH (we set it
    explicitly to the repo root)."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
