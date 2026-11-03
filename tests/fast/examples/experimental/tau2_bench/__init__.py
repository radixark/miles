"""Tests for the tau2-bench example, which lives outside this tree.

The example directory is ``tau2-bench`` -- not a Python identifier -- so
``EXAMPLE_DIR`` is what conftest puts on sys.path so its modules import by bare
name, as the rollout loads them.
"""

from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[5] / "examples" / "experimental" / "tau2-bench"
