"""Node diagnostic CLI E2E tests.

Runs on a live Ray cluster with GPU nodes. Tests both local (single-node)
and cluster (multi-node) modes of the node_diag CLI.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from miles.utils.ft.cli import app

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]

runner = CliRunner()


def test_local_checks(ray_cluster: None) -> None:
    """All local checks should pass on a healthy GPU node."""
    result = runner.invoke(app, ["diag", "local", "--timeout", "120"])
    assert result.exit_code == 0, f"local checks failed:\n{result.output}"
    for check_name in ["gpu", "nccl_simple", "disk", "network", "xid"]:
        assert check_name in result.output


def test_cluster_checks(ray_cluster: None) -> None:
    """All cluster checks should pass on a healthy Ray cluster."""
    result = runner.invoke(app, ["diag", "cluster", "--timeout", "180"])
    assert result.exit_code == 0, f"cluster checks failed:\n{result.output}"
    for check_name in ["gpu", "nccl_simple", "nccl_pairwise"]:
        assert check_name in result.output
