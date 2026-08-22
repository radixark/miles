from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH = REPO_ROOT / "docker" / "amd_patch" / "latest" / "miles.patch"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.rocm"


class TestTheRocmSourcePatch:
    def test_it_still_applies_to_the_sources_it_patches(self):
        """The rocm image applies this unconditionally, so a rename in miles/ breaks that build and nothing else."""
        result = subprocess.run(
            ["git", "apply", "--check", "--3way", "--unidiff-zero", str(PATCH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert "with conflicts" not in result.stderr, result.stderr
        assert result.returncode == 0, result.stderr

    def test_the_flags_here_are_the_flags_the_image_uses(self):
        """Checking the patch under other flags would prove something the build never does."""
        assert "git apply --3way --unidiff-zero /tmp/amd_patch/miles.patch" in DOCKERFILE.read_text()

    def test_every_method_it_calls_exists(self):
        """A 3-way apply can succeed with conflict markers, and the build only fails later, inside the actor."""
        source = (REPO_ROOT / "miles" / "backends" / "fsdp_utils" / "actor.py").read_text()
        added = [line for line in PATCH.read_text().splitlines() if line.startswith("+")]
        called = {
            line.split("self.")[1].split("(")[0] for line in added if "self." in line and "(" in line.split("self.")[1]
        }
        defined_by_the_patch = {line.split("def ")[1].split("(")[0] for line in added if "    def " in line}

        missing = sorted(name for name in called - defined_by_the_patch if f"def {name}(" not in source)

        assert missing == [], missing


@pytest.fixture(autouse=True)
def _require_a_git_checkout():
    assert (REPO_ROOT / ".git").exists(), "this test reads the patch against the working tree it lives in"
