"""The end of a job log must explain the failure, whatever the log's size.

A GPU suite writes tens of megabytes after the test that failed, so the
notifier's tail window never reaches the traceback. The suite runner already
sees every line, so it keeps the failing test's last output and repeats it in
the summary it prints last.
"""

import io
import logging
import sys
from collections import deque
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci
from tests.ci.ci_utils import FAILURE_TAIL_BYTES, FAILURE_TAIL_LINES
from tests.ci.ci_utils import TestFile as CITestFile
from tests.ci.ci_utils import _drain_child_output, _failure_tail, run_unittest_files

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# The window ci_failure_analysis downloads from the end of a job log.
NOTIFIER_TAIL_BYTES = 33_600


class _CapturedStdout:
    def __init__(self):
        self.buffer = io.BytesIO()


def drain(body: bytes) -> tuple[bytes, deque]:
    tail: deque = deque(maxlen=FAILURE_TAIL_LINES * 8)
    captured = _CapturedStdout()
    real, sys.stdout = sys.stdout, captured
    try:
        _drain_child_output(io.BytesIO(body), tail)
    finally:
        sys.stdout = real
    return captured.buffer.getvalue(), tail


def test_draining_passes_every_byte_through_untouched():
    body = b"".join(f"line {index}\n".encode() for index in range(50_000))
    passed_through, _ = drain(body)
    assert passed_through == body


def test_draining_holds_a_bounded_tail_of_a_large_stream():
    body = b"x" * 40_000_000
    _, tail = drain(body)
    assert sum(len(chunk) for chunk in tail) < 1_000_000


def test_the_tail_keeps_the_last_lines_and_drops_blank_ones():
    tail = deque([b"early\n", b"\n   \n", b"AssertionError: boom\n", b"exit 1\n"])
    assert _failure_tail(tail).splitlines() == ["early", "AssertionError: boom", "exit 1"]
    assert len(_failure_tail(deque([b"l\n" * 500])).splitlines()) <= FAILURE_TAIL_LINES


def test_a_failing_test_explains_itself_inside_the_notifier_window(tmp_path, monkeypatch, caplog):
    Path(tmp_path / "test_boom.py").write_text(
        "for index in range(5000):\n"
        "    print(f'noise {index}')\n"
        "raise AssertionError('the reactor exploded at step 42')\n"
    )
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        run_unittest_files([CITestFile(name="test_boom.py")], timeout_per_file=120, continue_on_error=True)
    assert "FAILED: test_boom.py returned exit code 1" in caplog.text
    _, separator, summary = caplog.text.rpartition("\nFAILED:\n")
    assert separator, "the runner printed no failure summary"
    assert "the reactor exploded at step 42" in summary
    assert len(summary) < NOTIFIER_TAIL_BYTES


def test_the_summary_survives_a_log_far_larger_than_the_notifier_window():
    body = b"".join(f"progress step {index}\n".encode() for index in range(400_000))
    body += b"AssertionError: the reactor exploded\n"
    _, tail = drain(body)
    rendered = _failure_tail(tail)
    assert "AssertionError: the reactor exploded" in rendered
    assert len(rendered.encode()) <= FAILURE_TAIL_BYTES
    assert len(body) > 100 * NOTIFIER_TAIL_BYTES
