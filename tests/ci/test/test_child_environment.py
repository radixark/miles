import io
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.ci import ci_utils
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


class TestChildEnvironment:
    @pytest.mark.parametrize("capture_output", [False, True])
    @pytest.mark.parametrize("environment", [None, {}, {"CHILD_ONLY": "explicit"}])
    @pytest.mark.parametrize("record_metrics", [False, True])
    def test_child_receives_an_isolated_environment_without_mutating_its_source(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capture_output: bool,
        environment: dict[str, str] | None,
        record_metrics: bool,
    ) -> None:
        """Explicit environments replace inheritance and metric directories affect only the child copy."""
        monkeypatch.setenv("PARENT_ONLY", "inherited")
        monkeypatch.delenv("MILES_SNAPSHOT_RECORD_DIR", raising=False)
        monkeypatch.delenv(ci_utils.CI_GATE_RECORD_DIR_ENV, raising=False)
        supplied = None if environment is None else environment.copy()
        before = dict(os.environ) if supplied is None else supplied.copy()
        record_dir = str(tmp_path / "metrics") if record_metrics else None
        observed: list[dict[str, str]] = []

        def child(*args: Any, env: dict[str, str], **kwargs: Any) -> SimpleNamespace:
            observed.append(env)
            return SimpleNamespace(
                stdout=io.StringIO("") if kwargs.get("text") else io.BytesIO(b""),
                returncode=0,
                wait=lambda: None,
            )

        def run_child(func: Callable[..., int], args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float) -> int:
            return func(args[0], capture_output=capture_output, record_dir=record_dir, env=supplied)

        monkeypatch.setattr(ci_utils.subprocess, "Popen", child)
        monkeypatch.setattr(ci_utils, "run_with_timeout", run_child)

        result = ci_utils.run_unittest_files(
            [ci_utils.TestFile(name="tests/fast/environment_probe.py", estimated_time=0)],
            timeout_per_file=10,
            enable_retry=False,
            continue_on_error=False,
        )

        assert result == 0
        assert len(observed) == 1
        expected = {**before, **({ci_utils.CI_GATE_RECORD_DIR_ENV: record_dir} if record_metrics else {})}
        assert observed[0] == expected
        assert observed[0] is not supplied
        assert (dict(os.environ) if supplied is None else supplied) == before
        assert (tmp_path / "metrics").exists() is record_metrics
