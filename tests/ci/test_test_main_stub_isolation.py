"""Regression test guarding the stub-isolation invariant of
``tests/fast/utils/debug_utils/run_megatron/worker/test_main.py``.

Contract under test (post-Round-4 architecture)
-----------------------------------------------
Round 4 (commit ``b19a8fd11``) moved ``test_main.py``'s ``sys.modules``
stubbing out of module-level code into a scoped pytest fixture
(``worker_main``) that uses ``monkeypatch.setitem`` so every mutation
is undone on test teardown.  The hard invariant after that refactor:

1. ``test_main.py``'s tests must pass cleanly in their own pytest
   session (no syntax / import-time issues).
2. Importing ``test_main.py`` as a plain module (without running its
   tests) MUST NOT install any stub for a
   ``miles.backends.megatron_utils.*`` leaf into ``sys.modules`` — i.e.
   if any such leaf appears in ``sys.modules`` after the import, it is
   a real on-disk module with a populated ``__file__`` or ``__path__``,
   never a fabricated ``ModuleType`` carrying only ``MagicMock``
   attributes.

This guards against a regression that took several rounds to pin
down: when stubbing was performed at module import time, the
under-populated stub for
``miles.backends.megatron_utils.checkpoint`` starved a downstream
sibling test (``test_lora_model_branches.py``) with
``ImportError: cannot import name 'save_checkpoint'``.  Re-introducing
module-level mutation would resurrect that failure mode.

The test runs everything in subprocesses so it never mutates the
parent interpreter's ``sys.modules``.  It depends only on stdlib and
pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# Leaf names that the old (pre-Round-4) ``test_main.py`` stubbed at
# module-import time.  The fixture-scoped architecture still references
# the same leaf set inside the fixture body, but it must NOT leak any
# of them into ``sys.modules`` at plain-import time.
_LEAF_NAMES = (
    "miles.backends.megatron_utils.arguments",
    "miles.backends.megatron_utils.checkpoint",
    "miles.backends.megatron_utils.initialize",
    "miles.backends.megatron_utils.model_provider",
)

_TEST_MAIN_RELPATH = "tests/fast/utils/debug_utils/run_megatron/worker/test_main.py"
_TEST_MAIN_DOTTED = "tests.fast.utils.debug_utils.run_megatron.worker.test_main"


_PROBE_SCRIPT = r"""
import json
import importlib
import sys

LEAF_NAMES = {leaf_names!r}
TARGET = {target!r}

# Snapshot which leaves are already in sys.modules *before* importing
# the test file, so we can attribute any newly-appeared stub to the
# import we are about to perform.
pre_import_present = {{name: name in sys.modules for name in LEAF_NAMES}}

try:
    importlib.import_module(TARGET)
    import_ok = True
    import_error = None
except BaseException as exc:
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
        "pre_import_present": pre_import_present[name],
        "in_sys_modules": mod is not None,
        "is_real_module": is_real_module,
        "file": getattr(mod, "__file__", None),
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


def _inherited_env(repo_root: Path) -> dict[str, str]:
    """Inherit the parent environment but force ``PYTHONPATH`` to the
    repo root so the child interpreter resolves ``tests.*`` and
    ``miles.*`` from the working tree."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    return env


def _run_test_main_in_subprocess(repo_root: Path) -> subprocess.CompletedProcess:
    """Run ``pytest -q`` on ``test_main.py`` in an isolated subprocess.

    Isolating the inner pytest session is the whole point: its
    ``sys.modules`` mutations (now fixture-scoped) live and die inside
    the child interpreter, never touching ours.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(repo_root / _TEST_MAIN_RELPATH),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=_inherited_env(repo_root),
    )


def _run_probe_in_subprocess(repo_root: Path) -> dict:
    """Import ``test_main.py`` as a plain module in a fresh
    subprocess and report the resulting ``sys.modules`` state for the
    four leaf names."""
    probe = _PROBE_SCRIPT.format(
        leaf_names=list(_LEAF_NAMES),
        target=_TEST_MAIN_DOTTED,
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=_inherited_env(repo_root),
    )
    assert result.returncode == 0, (
        f"probe subprocess failed (rc={result.returncode}).\n" f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    # The probe emits exactly one JSON line.  Tolerate trailing
    # whitespace or interleaved warnings by taking the last non-empty
    # stdout line.
    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert stdout_lines, f"probe produced no stdout. stderr:\n{result.stderr}"
    return json.loads(stdout_lines[-1])


def test_test_main_tests_pass_in_isolated_subprocess() -> None:
    """The whole point of Round 4's refactor: ``test_main.py``'s own
    tests must pass cleanly in an isolated pytest session.  If they
    don't, downstream invariants are moot."""
    repo_root = _repo_root()
    result = _run_test_main_in_subprocess(repo_root)
    assert result.returncode == 0, (
        "test_main.py's own tests failed in isolated subprocess "
        f"(rc={result.returncode}).\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_importing_test_main_does_not_install_starved_stubs() -> None:
    """Plain-import of ``test_main.py`` must not leak any starved stub
    into ``sys.modules`` for the four ``miles.backends.megatron_utils.*``
    leaves.

    Under the new fixture-scoped architecture, all stubbing happens
    inside the ``worker_main`` fixture via ``monkeypatch.setitem``, so
    a plain ``importlib.import_module`` of the test file must not
    mutate ``sys.modules`` at all for these leaves.  If a leaf does
    appear in ``sys.modules`` (e.g. because something else in the
    import chain pulled in a real implementation), it must be a real
    on-disk module — never a fabricated ``ModuleType`` with no
    ``__file__`` or ``__path__``.

    This catches any future regression that re-introduces module-level
    stub mutation (the original cause of the
    ``test_lora_model_branches.py`` starvation bug).
    """
    repo_root = _repo_root()
    payload = _run_probe_in_subprocess(repo_root)

    assert payload["import_ok"], "Plain import of test_main raised in probe subprocess:\n" f"{payload['import_error']}"

    records = payload["records"]
    starved = [r for r in records if r["in_sys_modules"] and not r["is_real_module"]]
    assert not starved, (
        "After plain-importing test_main.py, sys.modules contains "
        "starved stub(s) for miles.backends.megatron_utils leaves; "
        "this means stub mutation has leaked out of the worker_main "
        "fixture's monkeypatch scope and back into module-level code, "
        "resurrecting the architectural defect Round 4 fixed.\n"
        "Offending records:\n" + json.dumps(starved, indent=2)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
