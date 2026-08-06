import inspect
import logging
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
        # An empty ps listing is what a reap that worked looks like.
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(ci_utils, "_REAP_POLL_SECONDS", 0.0)
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

    def test_leftovers_that_outlive_the_wait_are_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A kill that missed leaves the next file on an occupied device, and staying silent
        about it is what makes the infrastructure failure read as that test being broken."""

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout="Sl   ray::MegatronTrainRayActor.train()\n")

        monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(ci_utils, "_REAP_POLL_SECONDS", 0.0)

        with caplog.at_level(logging.WARNING):
            ci_utils.reap_leaked_accelerator_processes()

        assert any("still alive" in record.message for record in caplog.records)

    def test_a_killed_process_awaiting_its_parent_is_not_mistaken_for_a_survivor(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SIGKILL leaves the entry in the table until the parent reaps it, and the job's pid 1
        never will, so counting those reports every successful reap as a failure."""

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout="Z    ray::MegatronTrainRayActor.train() <defunct>\n")

        monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(ci_utils, "_REAP_POLL_SECONDS", 0.0)

        with caplog.at_level(logging.WARNING):
            ci_utils.reap_leaked_accelerator_processes()

        assert not [record for record in caplog.records if "still alive" in record.message]

    def test_the_settle_window_is_spent_even_when_nothing_survives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The driver frees the memory after its holders are gone, so an empty process table is
        not yet a clean device; polling exists to tell whether the kill worked, not to cut the
        wait short."""
        slept: list[float] = []
        monkeypatch.setattr(
            ci_utils.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="")
        )
        monkeypatch.setattr(ci_utils.time, "sleep", slept.append)
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 4.0)

        ci_utils.reap_leaked_accelerator_processes()

        # Approximate because the survivor check itself consumes part of the window.
        assert sum(slept) == pytest.approx(4.0, abs=0.5)

    def test_a_clean_reap_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch, recorded_argvs: list[list[str]], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning only earns its place if the ordinary path stays quiet."""
        with caplog.at_level(logging.WARNING):
            ci_utils.reap_leaked_accelerator_processes()

        assert any(argv[0] == "ps" for argv in recorded_argvs)
        assert not [record for record in caplog.records if "still alive" in record.message]

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
