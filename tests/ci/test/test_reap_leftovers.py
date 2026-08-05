import inspect
import subprocess
import textwrap
from pathlib import Path

import pytest

from tests.ci import ci_utils, run_suite
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.fixture
def recorded_argvs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    argvs: list[list[str]] = []

    def fake_run(argv, **kwargs):
        argvs.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
    return argvs


@pytest.fixture
def one_passing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[ci_utils.TestFile]:
    """A single trivially-passing test file, so run_unittest_files really enters its per-file loop."""
    (tmp_path / "t_pass.py").write_text(
        textwrap.dedent(
            """
        import sys
        sys.exit(0)
    """
        )
    )
    monkeypatch.chdir(tmp_path)
    return [ci_utils.TestFile(name="t_pass.py", estimated_time=1)]


def _count_reaps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(ci_utils, "reap_leaked_accelerator_processes", lambda: calls.append("reaped"))
    return calls


class TestReapLeakedAcceleratorProcesses:
    def test_both_the_ray_runtime_and_the_leaked_engines_are_reaped(self, recorded_argvs: list[list[str]]) -> None:
        """A leaked sglang scheduler is not under ray's control, so stopping ray alone leaves it
        holding accelerator memory and the next test file starts on a dirty device."""
        ci_utils.reap_leaked_accelerator_processes()

        assert ["ray", "stop", "--force"] in recorded_argvs
        assert any("sglang::" in arg for argv in recorded_argvs for arg in argv)

    def test_the_engine_pattern_cannot_match_a_test_file_path(self, recorded_argvs: list[list[str]]) -> None:
        """Test paths under tests/e2e/sglang/ contain "sglang", so a bare pattern would make the
        reaper kill the very test process it is preparing for."""
        ci_utils.reap_leaked_accelerator_processes()

        patterns = [argv[-1] for argv in recorded_argvs if argv[0] == "pkill"]
        assert patterns
        for pattern in patterns:
            assert pattern.endswith("::")

    def test_running_test_files_does_not_reap_unless_asked(
        self, monkeypatch: pytest.MonkeyPatch, one_passing_file: list[ci_utils.TestFile]
    ) -> None:
        """Reaping is process-wide, so a run_unittest_files caller that is itself a test would be
        killed by its own reaper; only the CUDA suite runner opts in."""
        calls = _count_reaps(monkeypatch)

        ci_utils.run_unittest_files(one_passing_file, timeout_per_file=30)

        assert calls == []

    def test_running_test_files_reaps_before_a_file_when_asked(
        self, monkeypatch: pytest.MonkeyPatch, one_passing_file: list[ci_utils.TestFile]
    ) -> None:
        """Without this the CUDA suite is back to starting every file on whatever the previous one
        left holding the accelerators."""
        calls = _count_reaps(monkeypatch)

        ci_utils.run_unittest_files(one_passing_file, timeout_per_file=30, reap_leftovers=True)

        assert calls == ["reaped"]

    def test_the_cuda_suite_runner_asks_for_reaping(self) -> None:
        """The opt-in only protects CI if run_suite actually passes it."""
        source = inspect.getsource(run_suite.run_a_suite)

        assert "reap_leftovers=True" in source

    def test_a_missing_reaper_binary_does_not_abort_the_suite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reaping is best-effort housekeeping; letting OSError escape would fail every test file
        on a machine without pkill."""
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(
            ci_utils.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("pkill"))
        )

        ci_utils.reap_leaked_accelerator_processes()
