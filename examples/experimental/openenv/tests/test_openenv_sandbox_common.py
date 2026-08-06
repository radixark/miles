"""Offline unit tests for the shared sandbox layer (no network, no GPU).

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the adapter:

    pytest examples/experimental/openenv/tests/ -q

Covers the backend registry, whose whole job is to refuse to guess: which
provider runs decides whose quota a rollout spends, so an unnamed or unknown
one must fail at launch rather than resolve to whichever leg came first.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openenv_sandbox_common as common  # noqa: E402


@pytest.mark.parametrize(
    ("name", "expected"),
    [("daytona", "daytona"), ("e2b", "e2b"), ("agentenv", "e2b"), ("  E2B  ", "e2b")],
)
def test_resolve_backend_normalizes_names_and_aliases(name, expected):
    assert common.resolve_backend(name) == expected


@pytest.mark.parametrize("name", [None, "", "   "])
def test_resolve_backend_refuses_to_pick_for_you(name):
    with pytest.raises(ValueError, match="no sandbox backend named"):
        common.resolve_backend(name)


def test_resolve_backend_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown sandbox backend"):
        common.resolve_backend("modal")


def test_every_registered_backend_names_an_importable_target():
    """The registry is what the launcher passes to --custom-agent-function-path."""
    for backend, path in common.AGENT_FUNCTIONS.items():
        module, _, func = path.rpartition(".")
        assert func == "run", backend
        assert (Path(__file__).resolve().parent.parent / f"{module}.py").is_file(), backend
