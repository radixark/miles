import subprocess

import pytest

from tests.ci import ci_utils
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

    def test_a_missing_reaper_binary_does_not_abort_the_suite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reaping is best-effort housekeeping; letting OSError escape would fail every test file
        on a machine without pkill."""
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(
            ci_utils.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("pkill"))
        )

        ci_utils.reap_leaked_accelerator_processes()
