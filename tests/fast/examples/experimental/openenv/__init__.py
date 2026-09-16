"""Tests for the openenv tbench2 example, which lives outside this tree.

``EXAMPLE_DIR`` is the one place that spans the two: conftest puts it on
sys.path so the modules import by bare name, and the tests that assert on the
example's own files (an operator-facing help text, the agent-function targets a
launcher passes) read them from here.
"""

from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[5] / "examples" / "experimental" / "openenv"
