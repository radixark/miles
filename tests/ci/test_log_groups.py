"""Tests for the GitHub Actions log-group + per-attempt summary emission
added on top of `tests/ci/ci_utils.py:run_unittest_files`.

Two layers:

* Helper-level unit tests for `_gha_emit_group`, `_gha_emit_endgroup`,
  `_gha_emit_summary`: env gating and the title / summary encoding rules.
* Integration tests that drive `run_unittest_files` over real fake test
  scripts (`subprocess.Popen` is exercised end-to-end) and inspect the
  stdout stream byte-for-byte against the documented format.

The integration tests intentionally do not mock `run_with_timeout`,
`subprocess.Popen`, or any internal helper -- they only inject test files
and the env var, so that the full stdout / timeout / retry plumbing is
covered.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.ci_utils import TestFile, _gha_emit_endgroup, _gha_emit_group, _gha_emit_summary, run_unittest_files

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# ---------------------------------------------------------------------------
# Helpers used by integration tests
# ---------------------------------------------------------------------------


def _write_script(tmp: Path, name: str, body: str) -> str:
    """Write a Python script under ``tmp`` and return its bare filename.

    The bare filename is what gets passed into `TestFile(name=...)`;
    `run_unittest_files` joins it with `os.getcwd()` to obtain the real
    path, so callers must `monkeypatch.chdir(tmp)`.
    """
    path = tmp / name
    path.write_text(textwrap.dedent(body))
    return name


SUMMARY_LINE_RE = re.compile(
    r"^\[(?P<i>\d+)/(?P<n>\d+)\] (?P<filename>.+?)  (?P<status>PASS|FAIL|TIMEOUT)  elapsed=(?P<elapsed>\d+)s"
    r"(?: exit=(?P<exit>\d+))?"
    r"(?: timeout_after=(?P<timeout>\d+)s)?"
    r"(?: retry_of=attempt=(?P<retry_of>\d+))?$"
)


def _extract_marker_lines(stdout: str) -> list[str]:
    """Return only the workflow-command and summary lines, in order."""
    out: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("::group::") or line == "::endgroup::" or SUMMARY_LINE_RE.match(line):
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Helper-level unit tests: env gating
# ---------------------------------------------------------------------------


class TestEnvGate:
    def test_group_emitted_when_actions_true(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_group("hello")
        captured = capsys.readouterr()
        assert captured.out == "::group::hello\n"
        assert captured.err == ""

    def test_endgroup_emitted_when_actions_true(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_endgroup()
        captured = capsys.readouterr()
        assert captured.out == "::endgroup::\n"

    def test_summary_emitted_when_actions_true(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(1, 2, "tests/a.py", "PASS", 12.7)
        captured = capsys.readouterr()
        assert captured.out == "[1/2] tests/a.py  PASS  elapsed=12s\n"

    @pytest.mark.parametrize("value", ["", "false", "TRUE", "1", "yes"])
    def test_no_emission_when_actions_not_literal_true(self, monkeypatch, capsys, value):
        monkeypatch.setenv("GITHUB_ACTIONS", value)
        _gha_emit_group("hello")
        _gha_emit_endgroup()
        _gha_emit_summary(1, 2, "tests/a.py", "PASS", 12.7)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_emission_when_actions_unset(self, monkeypatch, capsys):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        _gha_emit_group("hello")
        _gha_emit_endgroup()
        _gha_emit_summary(1, 2, "tests/a.py", "PASS", 12.7)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Helper-level unit tests: title encoding (GHA workflow-command escapes)
# ---------------------------------------------------------------------------


class TestGroupTitleEncoding:
    def test_percent_encoded_first(self, monkeypatch, capsys):
        # `%` must be encoded BEFORE `\r` / `\n` to avoid double-encoding.
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_group("100% pass")
        captured = capsys.readouterr()
        assert captured.out == "::group::100%25 pass\n"

    def test_newline_encoded(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_group("line1\nline2")
        captured = capsys.readouterr()
        # Critical: must be a SINGLE workflow-command line; rendered \n
        # would break the command into two and fool the runner parser.
        assert captured.out == "::group::line1%0Aline2\n"
        assert captured.out.count("\n") == 1

    def test_carriage_return_encoded(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_group("part1\rpart2")
        captured = capsys.readouterr()
        assert captured.out == "::group::part1%0Dpart2\n"

    def test_percent_before_newline_combined(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_group("a%b\nc")
        captured = capsys.readouterr()
        # If `\n` were encoded first as `%0A`, then `%` -> `%25` would
        # rewrite it to `%250A` -> wrong. Correct order: `a%25b%0Ac`.
        assert captured.out == "::group::a%25b%0Ac\n"


# ---------------------------------------------------------------------------
# Helper-level unit tests: summary filename escape
# ---------------------------------------------------------------------------


class TestSummaryFilenameEscape:
    def test_newline_becomes_literal_backslash_n(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(1, 1, "weird\nname.py", "PASS", 5)
        captured = capsys.readouterr()
        # Two literal characters: backslash + 'n'. Verifies the summary
        # stays single-line under any filename pathology.
        assert captured.out == "[1/1] weird\\nname.py  PASS  elapsed=5s\n"
        assert captured.out.count("\n") == 1

    def test_carriage_return_becomes_literal_backslash_r(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(1, 1, "weird\rname.py", "PASS", 5)
        captured = capsys.readouterr()
        assert captured.out == "[1/1] weird\\rname.py  PASS  elapsed=5s\n"

    def test_percent_passes_through(self, monkeypatch, capsys):
        # Summary lines are normal log content, not workflow commands;
        # `%` is NOT escaped here (only \r/\n are).
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(1, 1, "100%name.py", "PASS", 5)
        captured = capsys.readouterr()
        assert captured.out == "[1/1] 100%name.py  PASS  elapsed=5s\n"


# ---------------------------------------------------------------------------
# Helper-level unit tests: summary suffix layout
# ---------------------------------------------------------------------------


class TestSummarySuffixOrder:
    def test_base_only(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(2, 5, "x.py", "PASS", 30.9)
        assert capsys.readouterr().out == "[2/5] x.py  PASS  elapsed=30s\n"

    def test_exit_suffix(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(2, 5, "x.py", "FAIL", 30.9, exit_code=1)
        assert capsys.readouterr().out == "[2/5] x.py  FAIL  elapsed=30s exit=1\n"

    def test_timeout_after_suffix(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(2, 5, "x.py", "TIMEOUT", 1800.0, timeout_after=1800)
        assert capsys.readouterr().out == "[2/5] x.py  TIMEOUT  elapsed=1800s timeout_after=1800s\n"

    def test_retry_of_suffix_alone(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(2, 5, "x.py", "PASS", 70.0, retry_of=1)
        assert capsys.readouterr().out == "[2/5] x.py  PASS  elapsed=70s retry_of=attempt=1\n"

    def test_timeout_then_retry_of(self, monkeypatch, capsys):
        # AC-fixed ordering: timeout_after BEFORE retry_of.
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(2, 5, "x.py", "TIMEOUT", 1800.0, timeout_after=1800, retry_of=1)
        assert capsys.readouterr().out == "[2/5] x.py  TIMEOUT  elapsed=1800s timeout_after=1800s retry_of=attempt=1\n"

    def test_exit_then_retry_of(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _gha_emit_summary(2, 5, "x.py", "FAIL", 130.0, exit_code=2, retry_of=1)
        assert capsys.readouterr().out == "[2/5] x.py  FAIL  elapsed=130s exit=2 retry_of=attempt=1\n"


# ---------------------------------------------------------------------------
# Integration test #1: PASS / FAIL(exit=1) / TIMEOUT in one run
# ---------------------------------------------------------------------------


class TestIntegrationStatusMatrix:
    def test_three_statuses_emit_complete_marker_quartet(self, tmp_path: Path, monkeypatch, capfd):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.chdir(tmp_path)

        _write_script(
            tmp_path,
            "t_pass.py",
            """
            import sys
            print("CHILD_PASS_OUT", flush=True)
            sys.exit(0)
            """,
        )
        _write_script(
            tmp_path,
            "t_fail.py",
            """
            import sys
            print("CHILD_FAIL_OUT", flush=True)
            sys.exit(1)
            """,
        )
        _write_script(
            tmp_path,
            "t_timeout.py",
            """
            import time
            print("CHILD_TIMEOUT_OUT", flush=True)
            time.sleep(60)
            """,
        )

        files = [
            TestFile(name="t_pass.py", estimated_time=1),
            TestFile(name="t_fail.py", estimated_time=1),
            TestFile(name="t_timeout.py", estimated_time=1),
        ]

        rc = run_unittest_files(
            files,
            timeout_per_file=2,
            continue_on_error=True,
            enable_retry=False,
        )
        assert rc == -1  # at least one failure -> overall fail

        out = capfd.readouterr().out
        lines = out.splitlines()

        # Each of the 3 files contributes exactly one group, one endgroup,
        # one summary line.
        group_lines = [ln for ln in lines if ln.startswith("::group::")]
        endgroup_lines = [ln for ln in lines if ln == "::endgroup::"]
        summary_lines = [ln for ln in lines if SUMMARY_LINE_RE.match(ln)]
        assert len(group_lines) == 3
        assert len(endgroup_lines) == 3
        assert len(summary_lines) == 3

        # Group title shape and ordering: 1-based index, attempt=1.
        assert group_lines[0].startswith("::group::t_pass.py  (1/3 est=1s attempt=1)")
        assert group_lines[1].startswith("::group::t_fail.py  (2/3 est=1s attempt=1)")
        assert group_lines[2].startswith("::group::t_timeout.py  (3/3 est=1s attempt=1)")

        # Summary content -- elapsed values vary, but format is fixed.
        # Use the regex to pull each one apart and assert fields.
        s0 = SUMMARY_LINE_RE.match(summary_lines[0]).groupdict()
        s1 = SUMMARY_LINE_RE.match(summary_lines[1]).groupdict()
        s2 = SUMMARY_LINE_RE.match(summary_lines[2]).groupdict()
        assert (s0["i"], s0["n"], s0["filename"], s0["status"], s0["exit"], s0["timeout"], s0["retry_of"]) == (
            "1",
            "3",
            "t_pass.py",
            "PASS",
            None,
            None,
            None,
        )
        assert (s1["i"], s1["n"], s1["filename"], s1["status"], s1["exit"], s1["timeout"], s1["retry_of"]) == (
            "2",
            "3",
            "t_fail.py",
            "FAIL",
            "1",
            None,
            None,
        )
        assert (s2["i"], s2["n"], s2["filename"], s2["status"], s2["exit"], s2["timeout"], s2["retry_of"]) == (
            "3",
            "3",
            "t_timeout.py",
            "TIMEOUT",
            None,
            "2",
            None,
        )

        # Marker ordering per file: group -> child-stdout -> endgroup -> summary.
        # Child stdout strings should appear BETWEEN their group and endgroup.
        def _idx(token: str) -> int:
            for k, ln in enumerate(lines):
                if token in ln:
                    return k
            raise AssertionError(f"token {token!r} not found in stdout")

        for child_token, group_idx, end_idx in [
            ("CHILD_PASS_OUT", 0, 0),
            ("CHILD_FAIL_OUT", 1, 1),
            ("CHILD_TIMEOUT_OUT", 2, 2),
        ]:
            g = lines.index(group_lines[group_idx])
            e = lines.index(endgroup_lines[end_idx], g)
            c = _idx(child_token)
            s = lines.index(summary_lines[group_idx], e)
            assert g < c < e < s, f"order broken for {child_token!r}: group={g} child={c} endgroup={e} summary={s}"


# ---------------------------------------------------------------------------
# Integration test #2: TIMEOUT closes its group + emits expected summary
# ---------------------------------------------------------------------------


class TestIntegrationTimeoutClosesGroup:
    def test_timeout_summary_includes_timeout_after(self, tmp_path: Path, monkeypatch, capfd):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.chdir(tmp_path)

        _write_script(
            tmp_path,
            "t_hang.py",
            """
            import time
            time.sleep(60)
            """,
        )

        files = [TestFile(name="t_hang.py", estimated_time=1)]

        rc = run_unittest_files(
            files,
            timeout_per_file=2,
            continue_on_error=False,
            enable_retry=False,
        )
        assert rc == -1

        out = capfd.readouterr().out
        lines = out.splitlines()
        group_lines = [ln for ln in lines if ln.startswith("::group::")]
        endgroup_lines = [ln for ln in lines if ln == "::endgroup::"]
        summary_lines = [ln for ln in lines if SUMMARY_LINE_RE.match(ln)]

        assert len(group_lines) == 1
        assert len(endgroup_lines) == 1, "TIMEOUT must still emit endgroup (group balance)"
        assert len(summary_lines) == 1

        m = SUMMARY_LINE_RE.match(summary_lines[0]).groupdict()
        assert m["status"] == "TIMEOUT"
        assert m["timeout"] == "2"  # effective_timeout = max(2, int(1*1.25)) = 2
        assert m["exit"] is None
        assert m["retry_of"] is None


# ---------------------------------------------------------------------------
# Integration test #3: retry FAIL -> PASS path
# ---------------------------------------------------------------------------


class TestIntegrationRetryFailThenPass:
    def test_retry_emits_two_complete_marker_triples(self, tmp_path: Path, monkeypatch, capfd):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.chdir(tmp_path)

        state_file = tmp_path / "retry_state.txt"
        # Script reads state file: first call writes "1" then exits 1 with
        # a retriable output marker; second call sees "1" and exits 0.
        script_body = f"""
            import os, sys
            state_path = {str(state_file)!r}
            if os.path.exists(state_path):
                # second attempt: PASS
                print("CHILD_RETRY_PASS", flush=True)
                sys.exit(0)
            else:
                with open(state_path, "w") as f:
                    f.write("1")
                # First attempt: print retriable-pattern keyword so
                # is_retriable_failure() returns True, then exit non-zero.
                print("CHILD_RETRY_FAIL: accuracy regression", flush=True)
                sys.exit(1)
            """

        _write_script(tmp_path, "t_retry.py", script_body)

        files = [TestFile(name="t_retry.py", estimated_time=1)]

        rc = run_unittest_files(
            files,
            timeout_per_file=10,
            continue_on_error=False,
            enable_retry=True,
            max_attempts=2,
            retry_wait_seconds=0,  # do not delay the test
        )
        assert rc == 0

        out = capfd.readouterr().out
        lines = out.splitlines()
        group_lines = [ln for ln in lines if ln.startswith("::group::")]
        endgroup_lines = [ln for ln in lines if ln == "::endgroup::"]
        summary_lines = [ln for ln in lines if SUMMARY_LINE_RE.match(ln)]

        assert len(group_lines) == 2
        assert len(endgroup_lines) == 2
        assert len(summary_lines) == 2

        # Group titles carry attempt=1 and attempt=2.
        assert "attempt=1)" in group_lines[0]
        assert "attempt=2)" in group_lines[1]

        m1 = SUMMARY_LINE_RE.match(summary_lines[0]).groupdict()
        m2 = SUMMARY_LINE_RE.match(summary_lines[1]).groupdict()
        assert m1["status"] == "FAIL"
        assert m1["exit"] == "1"
        assert m1["retry_of"] is None
        assert m2["status"] == "PASS"
        assert m2["exit"] is None
        assert m2["timeout"] is None
        assert m2["retry_of"] == "1"


# ---------------------------------------------------------------------------
# Integration test: baseline (GITHUB_ACTIONS unset = byte-for-byte unchanged)
# ---------------------------------------------------------------------------


class TestIntegrationBaselineLocalMode:
    def test_no_marker_lines_in_stdout_when_actions_unset(self, tmp_path: Path, monkeypatch, capfd):
        # Whatever non-`true` value GITHUB_ACTIONS holds, none of the
        # marker emit helpers should produce output, and parent-side
        # stdout must therefore contain only what was already there
        # (here: just child stdout in inherit mode).
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.chdir(tmp_path)

        _write_script(
            tmp_path,
            "t_a.py",
            """
            print("BASELINE_A", flush=True)
            """,
        )
        _write_script(
            tmp_path,
            "t_b.py",
            """
            print("BASELINE_B", flush=True)
            """,
        )

        files = [
            TestFile(name="t_a.py", estimated_time=1),
            TestFile(name="t_b.py", estimated_time=1),
        ]

        rc = run_unittest_files(
            files,
            timeout_per_file=10,
            continue_on_error=True,
            enable_retry=False,
        )
        assert rc == 0

        out = capfd.readouterr().out
        # No workflow-command markers anywhere.
        assert "::group::" not in out
        assert "::endgroup::" not in out
        # No `[i/N] ... STATUS ... elapsed=...s` summary lines.
        for line in out.splitlines():
            assert not SUMMARY_LINE_RE.match(line), f"unexpected summary line in non-CI mode: {line!r}"
        # Child stdout itself is preserved (inherit mode).
        assert "BASELINE_A" in out
        assert "BASELINE_B" in out

    def test_explicit_false_also_suppresses_markers(self, tmp_path: Path, monkeypatch, capfd):
        monkeypatch.setenv("GITHUB_ACTIONS", "false")
        monkeypatch.chdir(tmp_path)

        _write_script(
            tmp_path,
            "t_a.py",
            """
            print("FALSE_MODE_A", flush=True)
            """,
        )
        files = [TestFile(name="t_a.py", estimated_time=1)]

        rc = run_unittest_files(
            files,
            timeout_per_file=10,
            continue_on_error=True,
            enable_retry=False,
        )
        assert rc == 0

        out = capfd.readouterr().out
        assert "::group::" not in out
        assert "::endgroup::" not in out
        assert "FALSE_MODE_A" in out
