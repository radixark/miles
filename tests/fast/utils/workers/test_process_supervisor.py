from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

import pytest
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.api import RunProcsResult, SignalException
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure

import miles.utils.workers.process_supervisor as supervisor_module
from miles.utils.workers.process_supervisor import (
    _DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS,
    _SIGNALS_TO_HANDLE_ENV_VAR,
    SUBPROCESS_INDEX_ENV_VAR,
    _exit_code_from_result,
    _exit_code_from_wait_result,
    _report_signal_received_during_teardown,
    main,
    supervise,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CHILD_SCRIPT = Path(__file__).resolve()
SUPERVISOR_EXIT_TIMEOUT_SECONDS = 30.0
SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS = 10.0
GRACE_PERIOD_SECONDS = 0.5
GRACE_PERIOD_UPPER_BOUND_SLACK_SECONDS = 10.0
DOUBLE_SIGNAL_GRACE_PERIOD_SECONDS = 3.0
SECOND_SIGNAL_DELAY_SECONDS = 0.5
DOCUMENTED_SUBPROCESS_INDEX_ENV_VAR = "MILES_SUPERVISOR_SUBPROCESS_INDEX"
DOCUMENTED_DEFAULT_GRACE_PERIOD_SECONDS = 20.0
KUBERNETES_DEFAULT_GRACE_PERIOD_SECONDS = 30.0
CHILD_PID = 4002
OTHER_CHILD_PID = CHILD_PID + 1
RECORDED_SUPERVISOR_EXIT_CODE = 17

ENV_MARKER_VAR = "MILES_TEST_ENV_MARKER"
STDOUT_MARKER = "child-stdout-"
STDERR_MARKER = "child-stderr-"

MODE_SLEEP = "sleep"
MODE_EXIT = "exit"
MODE_SUICIDE = "suicide"
MODE_IGNORE_SIGTERM = "ignore-sigterm"
MODE_RECORD_SIGNAL = "record-signal"

REPORT_WAIT_TIMEOUT_SECONDS = 30.0
_GRANDCHILD = "import sys, time\nopen(sys.argv[1], 'w').close()\nwhile True: time.sleep(3600)"


@pytest.mark.usefixtures("unconfigured_root_logging")
class TestArgumentValidation:
    def test_a_missing_separator_is_rejected(self, capsys):
        """The supervised command must be introduced by '--', even when the flags parse cleanly."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "2"])

        assert exc_info.value.code == 2
        assert "must be given after '--'" in capsys.readouterr().err

    def test_a_command_given_without_the_separator_is_rejected(self):
        """A command written without '--' would otherwise be parsed as stray flags."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "2", "sleep", "1"])

        assert exc_info.value.code == 2

    def test_an_empty_command_is_rejected(self):
        """A '--' with nothing after it is an error rather than a supervisor with nothing to run."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "2", "--"])

        assert exc_info.value.code == 2

    @pytest.mark.parametrize("count", ["0", "-1", "-8"])
    def test_a_non_positive_subprocess_count_is_rejected(self, count):
        """Supervising zero or fewer subprocesses is meaningless and is rejected up front."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", count, "--", "sleep", "1"])

        assert exc_info.value.code == 2

    @pytest.mark.parametrize("grace_period", ["inf", "-inf", "nan"])
    def test_a_non_finite_grace_period_is_rejected(self, grace_period):
        """A non-finite deadline would either disable the SIGKILL escalation or never compare true."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "1", "--termination-grace-period-seconds", grace_period, "--", "sleep", "1"])

        assert exc_info.value.code == 2

    @pytest.mark.parametrize("grace_period", ["-1", "-0.001", "-3600"])
    def test_a_negative_grace_period_is_rejected(self, grace_period):
        """A negative grace period is a typo rather than a request to SIGKILL immediately."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "1", "--termination-grace-period-seconds", grace_period, "--", "sleep", "1"])

        assert exc_info.value.code == 2

    def test_a_missing_subprocess_count_is_rejected(self):
        """There is no sensible default for how many ranks the pod holds."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--", "sleep", "1"])

        assert exc_info.value.code == 2

    def test_a_non_numeric_subprocess_count_is_rejected(self):
        """A count that is not an integer fails at parse time rather than at spawn time."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "two", "--", "sleep", "1"])

        assert exc_info.value.code == 2

    def test_help_is_printed_without_a_supervised_command(self, capsys):
        """Asking for help must not require inventing a command to supervise."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        assert SUBPROCESS_INDEX_ENV_VAR in capsys.readouterr().out

    def test_a_zero_grace_period_is_accepted(self, monkeypatch):
        """Escalating immediately is a legitimate request, unlike a negative period."""
        created = _record_supervisor(monkeypatch)

        main(["--num-subprocesses", "1", "--termination-grace-period-seconds", "0", "--", "sleep", "1"])

        assert created[0]["termination_grace_period_seconds"] == 0.0


@pytest.mark.usefixtures("unconfigured_root_logging")
class TestMain:
    def test_the_parsed_arguments_reach_the_supervisor_and_its_code_is_returned(self, monkeypatch):
        """main is a thin wrapper: it parses, constructs, and propagates the exit code."""
        created = _record_supervisor(monkeypatch)

        exit_code = main(["--num-subprocesses", "2", "--termination-grace-period-seconds", "1.5", "--", "sleep", "1"])

        assert exit_code == RECORDED_SUPERVISOR_EXIT_CODE
        assert created == [{"command": ["sleep", "1"], "num_subprocesses": 2, "termination_grace_period_seconds": 1.5}]

    def test_a_separator_inside_the_supervised_command_is_left_alone(self, monkeypatch):
        """Only the first '--' splits, so the supervised command may contain its own."""
        created = _record_supervisor(monkeypatch)

        main(["--num-subprocesses", "1", "--", "bash", "-c", "--", "payload"])

        assert created[0]["command"] == ["bash", "-c", "--", "payload"]

    def test_the_grace_period_defaults_when_the_flag_is_absent(self, monkeypatch):
        """The default must expire before the Kubernetes termination grace period it runs inside."""
        created = _record_supervisor(monkeypatch)

        main(["--num-subprocesses", "1", "--", "sleep", "1"])

        assert created[0]["termination_grace_period_seconds"] == DOCUMENTED_DEFAULT_GRACE_PERIOD_SECONDS

    def test_the_process_arguments_are_used_when_no_argv_is_given(self, monkeypatch):
        """The container entrypoint calls main with no arguments at all."""
        created = _record_supervisor(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["process_supervisor", "--num-subprocesses", "1", "--", "sleep", "1"])

        main()

        assert created[0]["command"] == ["sleep", "1"]

    def test_an_explicitly_empty_argv_is_not_confused_with_no_argv(self, monkeypatch):
        """Falling back to sys.argv here would make main run whatever the caller was invoked with."""
        monkeypatch.setattr(sys, "argv", ["process_supervisor", "--num-subprocesses", "1", "--", "sleep", "1"])

        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code == 2

    def test_the_function_default_grace_period_matches_the_documented_one(self):
        """Callers that call supervise directly must get the same default as the CLI."""
        parameters = inspect.signature(supervise).parameters

        assert parameters["termination_grace_period_seconds"].default == DOCUMENTED_DEFAULT_GRACE_PERIOD_SECONDS


class TestExitCodeFromWaitResult:
    def test_a_normal_exit_keeps_its_code(self):
        """Popen reports a normal exit as the exit code itself."""
        assert _exit_code_from_wait_result(7) == 7

    def test_a_successful_exit_stays_zero(self):
        """Zero must not be mistaken for a signalled death."""
        assert _exit_code_from_wait_result(0) == 0

    def test_the_highest_exit_code_is_left_alone(self):
        """Exit code 255 is the top of the range and must not be translated."""
        assert _exit_code_from_wait_result(255) == 255

    @pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGTERM, signal.SIGINT, signal.SIGABRT])
    def test_a_signalled_exit_becomes_128_plus_that_signal(self, signum):
        """Popen reports a signalled death as a negative signal number, which no shell understands."""
        assert _exit_code_from_wait_result(-int(signum)) == 128 + signum


class TestExitCodeFromResult:
    def test_a_run_without_failures_is_a_success(self):
        """A batch where every rank exited zero produces no failures at all."""
        assert _exit_code_from_result(_run_result()) == 0

    def test_the_failure_torchelastic_recorded_first_decides(self):
        """torchelastic records the rank that actually failed before the peers it then tears down."""
        failures = {
            1: _failure(local_rank=1, exitcode=7, timestamp=20),
            0: _failure(exitcode=9, timestamp=10),
        }
        assert _exit_code_from_result(_run_result(failures)) == 7

    def test_a_signalled_cause_beats_a_peer_that_traps_the_teardown_signal(self):
        """A peer turning SIGTERM into a plain exit code ties on the second-resolution timestamp."""
        failures = {
            0: _failure(exitcode=-int(signal.SIGSEGV), timestamp=10),
            1: _failure(local_rank=1, exitcode=42, timestamp=10),
        }
        assert _exit_code_from_result(_run_result(failures)) == 128 + signal.SIGSEGV

    def test_a_signalled_earliest_failure_maps_to_128_plus_the_signal(self):
        """A rank dying of SIGKILL surfaces as 137, the shell convention."""
        failures = {0: _failure(exitcode=-int(signal.SIGKILL), timestamp=10)}
        assert _exit_code_from_result(_run_result(failures)) == 128 + signal.SIGKILL

    def test_the_highest_exit_code_is_reported_unchanged(self):
        """Exit code 255 must not be masked down into the low seven bits."""
        failures = {0: _failure(exitcode=255, timestamp=10)}
        assert _exit_code_from_result(_run_result(failures)) == 255


class TestSpawning:
    @pytest.mark.parametrize("count", [0, -1])
    def test_a_non_positive_subprocess_count_is_refused(self, count):
        """A count derived from an empty config would otherwise report success without running anything."""
        with pytest.raises(ValueError):
            supervise(command=["sleep", "1"], num_subprocesses=count)

    def test_every_subprocess_is_started_with_its_own_index(self, monkeypatch):
        """Each rank has to be told which of the N copies it is, through the environment."""
        started = _record_start_processes(monkeypatch)

        supervise(command=["sleep", "1"], num_subprocesses=3)

        assert started[0]["envs"] == {
            0: {SUBPROCESS_INDEX_ENV_VAR: "0"},
            1: {SUBPROCESS_INDEX_ENV_VAR: "1"},
            2: {SUBPROCESS_INDEX_ENV_VAR: "2"},
        }

    def test_the_supervised_command_is_spawned_directly(self, monkeypatch):
        """No wrapper sits between torchelastic and the command, so the pid it logs is the command's."""
        started = _record_start_processes(monkeypatch)

        supervise(command=["sleep", "1"], num_subprocesses=2)

        assert started[0]["entrypoint"] == "sleep"
        assert started[0]["args"] == {0: ("1",), 1: ("1",)}

    def test_both_standard_streams_are_teed_back_with_rank_prefixes(self, monkeypatch):
        """The pod log interleaves every rank's output, so each line must say which rank wrote it."""
        started = _record_start_processes(monkeypatch)

        supervise(command=["sleep", "1"], num_subprocesses=2)

        assert isinstance(started[0]["logs_specs"], DefaultLogsSpecs)
        assert started[0]["log_line_prefixes"] == {0: "[rank0] ", 1: "[rank1] "}
        destinations = started[0]["logs_specs"].reify({0: {}, 1: {}})
        assert sorted(destinations.tee_stdouts) == [0, 1]
        assert sorted(destinations.tee_stderrs) == [0, 1]

    def test_the_signals_torchelastic_handles_are_pinned_before_it_reads_them(self, monkeypatch):
        """torchelastic reads this variable while spawning, so pinning it afterwards would be a no-op."""
        started = _record_start_processes(monkeypatch)
        monkeypatch.setenv(_SIGNALS_TO_HANDLE_ENV_VAR, "SIGHUP")

        supervise(command=["sleep", "1"], num_subprocesses=1)

        assert started[0]["signals_to_handle"] == "SIGTERM,SIGINT"

    def test_each_spawned_subprocess_is_logged_under_its_own_index(self, monkeypatch, caplog):
        """An operator maps a rank to a pid using this line and nothing else."""
        context = _fake_context(num_subprocesses=2, wait_outcome=_run_result())

        with caplog.at_level(logging.INFO, logger=supervisor_module.__name__):
            _run(monkeypatch, context=context, num_subprocesses=2)

        assert f"Spawned subprocess index=0 pid={CHILD_PID}" in caplog.text
        assert f"Spawned subprocess index=1 pid={OTHER_CHILD_PID}" in caplog.text


class TestSubprocessExit:
    def test_every_subprocess_exiting_cleanly_yields_zero(self, monkeypatch):
        """torchelastic waits for the whole batch, and a clean batch is a clean pod."""
        context = _fake_context(wait_outcome=_run_result())

        assert _run(monkeypatch, context=context) == 0

    def test_the_exit_code_of_the_earliest_failure_is_returned(self, monkeypatch):
        """The rank that died first is the cause; the ones torn down after it are consequences."""
        failures = {
            1: _failure(local_rank=1, exitcode=7, timestamp=10),
            0: _failure(exitcode=-int(signal.SIGTERM), timestamp=10),
        }
        context = _fake_context(num_subprocesses=2, wait_outcome=_run_result(failures))

        assert _run(monkeypatch, context=context, num_subprocesses=2) == 7

    def test_a_subprocess_killed_by_a_signal_is_reported_as_128_plus_the_signal(self, monkeypatch):
        """Popen reports a signalled death as a negative number, which no shell would understand."""
        context = _fake_context(wait_outcome=_run_result({0: _failure(exitcode=-int(signal.SIGKILL))}))

        assert _run(monkeypatch, context=context) == 128 + signal.SIGKILL

    def test_an_unexpected_failure_tears_the_batch_down_before_propagating(self, monkeypatch, caplog):
        """Leaving the tee threads running would hang the container instead of failing the pod."""
        context = _fake_context(wait_outcome=RuntimeError("torchelastic bookkeeping blew up"))

        with caplog.at_level(logging.ERROR, logger=supervisor_module.__name__), pytest.raises(RuntimeError):
            _run(monkeypatch, context=context, termination_grace_period_seconds=2.0)

        assert context.close_calls == [(signal.SIGTERM, 2.0)]
        assert any(record.exc_info is not None for record in caplog.records)

    def test_a_failure_that_is_not_an_exception_still_tears_the_batch_down(self, monkeypatch):
        """SystemExit and KeyboardInterrupt skip an `except Exception`, leaving the tee threads alive."""
        context = _fake_context(wait_outcome=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            _run(monkeypatch, context=context, termination_grace_period_seconds=2.0)

        assert context.close_calls == [(signal.SIGTERM, 2.0)]

    def test_every_failure_is_logged_with_its_index_and_pid(self, monkeypatch, caplog):
        """An operator reads these lines to find out which rank went down and how."""
        context = _fake_context(wait_outcome=_run_result({0: _failure(exitcode=7)}))

        with caplog.at_level(logging.WARNING, logger=supervisor_module.__name__):
            _run(monkeypatch, context=context)

        assert f"Subprocess index=0 pid={CHILD_PID} exited with 7" in caplog.text


class TestSignalForwarding:
    def test_a_forwarded_signal_decides_the_exit_code(self, monkeypatch):
        """A pod terminated by the kubelet exits the way a shell would report the signal."""
        context = _fake_context(wait_outcome=SignalException("got SIGTERM", sigval=signal.SIGTERM))

        assert _run(monkeypatch, context=context) == 128 + signal.SIGTERM

    def test_the_received_signal_and_the_grace_period_shape_the_teardown(self, monkeypatch):
        """Forwarding SIGTERM for a SIGINT would deny the subprocesses their own Ctrl-C handling."""
        context = _fake_context(wait_outcome=SignalException("got SIGINT", sigval=signal.SIGINT))

        _run(monkeypatch, context=context, termination_grace_period_seconds=2.0)

        assert context.close_calls == [(signal.SIGINT, 2.0)]

    def test_further_forwarded_signals_are_reported_instead_of_interrupting(self, monkeypatch):
        """torchelastic leaves its handler installed, so a repeated signal would raise inside close."""
        context = _fake_context(wait_outcome=SignalException("got SIGTERM", sigval=signal.SIGTERM))

        _run(monkeypatch, context=context)

        installed = context.handlers_during_close[0]
        assert installed == (_report_signal_received_during_teardown, _report_signal_received_during_teardown)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            installed[0](int(signal.SIGINT), None)

    def test_the_repeat_notice_reaches_stderr_without_taking_a_lock(self, capfd):
        """A handler that goes through logging drops the line, or nests, when signals arrive in bursts."""
        source = inspect.getsource(_report_signal_received_during_teardown)

        _report_signal_received_during_teardown(int(signal.SIGTERM), None)

        assert "SIGTERM while already tearing down" in capfd.readouterr().err
        assert "logger" not in source

    def test_the_handlers_are_replaced_before_anything_else_in_the_teardown(self, monkeypatch):
        """Every instruction between torchelastic raising and the swap is exposed to a second signal."""
        context = _fake_context(wait_outcome=SignalException("got SIGTERM", sigval=signal.SIGTERM))
        handlers_when_logging = []

        def record(message, *args) -> None:
            handlers_when_logging.append(signal.getsignal(signal.SIGTERM))

        monkeypatch.setattr(supervisor_module.logger, "info", record)

        _run(monkeypatch, context=context)

        assert handlers_when_logging[-1] is _report_signal_received_during_teardown

    def test_a_signal_arriving_while_spawning_still_decides_the_exit_code(self, monkeypatch, caplog):
        """torchelastic installs its handler before the last rank is spawned, so the signal lands here."""

        def start_processes(**kwargs) -> object:
            raise SignalException("got SIGTERM", sigval=signal.SIGTERM)

        monkeypatch.setattr(supervisor_module, "start_processes", start_processes)

        with caplog.at_level(logging.INFO, logger=supervisor_module.__name__):
            assert supervise(command=["true"], num_subprocesses=2) == 128 + signal.SIGTERM

        assert "Received SIGTERM while still spawning" in caplog.text
        assert "forwarding it to every subprocess" not in caplog.text

    def test_the_forwarding_is_announced_in_the_log(self, monkeypatch, caplog):
        """An operator reconstructs why a pod died from this line."""
        context = _fake_context(wait_outcome=SignalException("got SIGTERM", sigval=signal.SIGTERM))

        with caplog.at_level(logging.INFO, logger=supervisor_module.__name__):
            _run(monkeypatch, context=context)

        assert "Received SIGTERM, forwarding it to every subprocess" in caplog.text


def _run(
    monkeypatch,
    *,
    context,
    num_subprocesses: int = 1,
    termination_grace_period_seconds: float = 1.0,
) -> int:
    monkeypatch.setattr(supervisor_module, "start_processes", lambda **kwargs: context)
    return supervise(
        command=["true"],
        num_subprocesses=num_subprocesses,
        termination_grace_period_seconds=termination_grace_period_seconds,
    )


def _record_start_processes(monkeypatch) -> list[dict]:
    started: list[dict] = []

    def start_processes(**kwargs) -> object:
        started.append({**kwargs, "signals_to_handle": os.environ.get(_SIGNALS_TO_HANDLE_ENV_VAR)})
        return _fake_context(num_subprocesses=len(kwargs["envs"]), wait_outcome=_run_result())

    monkeypatch.setattr(supervisor_module, "start_processes", start_processes)
    return started


class TestHappyPath:
    def test_every_copy_runs_to_completion_and_the_supervisor_exits_zero(self, launch_supervisor):
        """Each copy sees its own index, finishes with exit code 0, and the supervisor reports success."""
        run = launch_supervisor(
            num_subprocesses=3,
            child_args=["--mode", "exit", "--exit-code", "0", "--wait-for-reports", "3"],
        )

        reports = run.wait_for_children(3)

        assert run.wait_for_exit() == 0
        assert sorted(reports) == ["0", "1", "2"]
        assert _modes(reports) == {"0": "exit", "1": "exit", "2": "exit"}


class TestSubprocessLaunching:
    def test_every_subprocess_gets_its_own_index_and_session(self, launch_supervisor):
        """Every index in 0..N-1 shows up exactly once, on a distinct pid that leads its own session."""
        run = launch_supervisor(num_subprocesses=3, child_args=["--mode", "sleep"])

        reports = run.wait_for_children(3)

        assert sorted(reports) == ["0", "1", "2"]
        assert len({report["pid"] for report in reports.values()}) == 3
        assert all(report["process_group_id"] == report["pid"] for report in reports.values())
        assert all(report["session_id"] == report["pid"] for report in reports.values())

    def test_the_external_contract_stays_where_consumers_expect_it(self):
        """Both values are part of the pod contract, so drifting from them is a breaking change."""
        assert SUBPROCESS_INDEX_ENV_VAR == DOCUMENTED_SUBPROCESS_INDEX_ENV_VAR
        assert _DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS == DOCUMENTED_DEFAULT_GRACE_PERIOD_SECONDS
        assert _DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS < KUBERNETES_DEFAULT_GRACE_PERIOD_SECONDS

    def test_an_index_inherited_from_the_parent_is_overwritten(self, launch_supervisor):
        """A stale index left in the pod environment must not leak into any subprocess."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "sleep"],
            extra_env={DOCUMENTED_SUBPROCESS_INDEX_ENV_VAR: "poison"},
        )

        reports = run.wait_for_children(2)

        assert sorted(reports) == ["0", "1"]

    def test_both_standard_streams_reach_the_container_log_with_rank_prefixes(self, launch_supervisor):
        """Container logs interleave every rank's streams, so each line carries the rank that wrote it."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "exit", "--emit-stream-markers", "--wait-for-reports", "2"],
            capture_output=True,
        )

        stdout, stderr = run.process.communicate(timeout=SUPERVISOR_EXIT_TIMEOUT_SECONDS)

        for index in ("0", "1"):
            assert f"[rank{index}] {STDOUT_MARKER}{index}" in stdout
            assert f"[rank{index}] {STDERR_MARKER}{index}" in stderr

    def test_a_single_subprocess_is_supervised_like_any_other_count(self, launch_supervisor):
        """The degenerate one-rank-per-pod case still spawns, supervises and propagates."""
        run = launch_supervisor(
            num_subprocesses=1,
            child_args=["--mode", "exit", "--exit-code", "5"],
        )

        assert run.wait_for_exit() == 5
        assert sorted(run.reports()) == ["0"]

    def test_the_supervisor_environment_reaches_every_subprocess(self, launch_supervisor):
        """The index var is added to the inherited environment rather than replacing it."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "sleep"],
            extra_env={ENV_MARKER_VAR: "inherited"},
        )

        reports = run.wait_for_children(2)

        assert [report["env_marker"] for report in reports.values()] == ["inherited", "inherited"]

    def test_keeps_running_while_every_subprocess_is_alive(self, launch_supervisor):
        """The supervisor does not exit on its own as long as no subprocess has exited."""
        run = launch_supervisor(num_subprocesses=2, child_args=["--mode", "sleep"])
        run.wait_for_children(2)

        time.sleep(0.5)

        assert run.process.poll() is None

    def test_a_command_that_cannot_be_spawned_at_all_exits_nonzero(self, launch_supervisor):
        """A typo in the entrypoint must fail the pod immediately rather than hang it."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=[],
            command=[str(REPO_ROOT / "definitely-not-an-executable")],
        )

        assert run.wait_for_exit() == 1
        assert run.reports() == {}

    def test_the_container_log_is_configured_to_be_readable(self, monkeypatch):
        """These lines are interleaved with every rank's own output, so they must stay identifiable."""
        configured: list[dict] = []
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: configured.append(kwargs))
        _record_supervisor(monkeypatch)

        main(["--num-subprocesses", "1", "--", "sleep", "1"])

        assert [kwargs["level"] for kwargs in configured] == [logging.INFO]
        assert all(part in configured[0]["format"] for part in ("%(asctime)s", "process_supervisor", "%(levelname)s"))


class TestSupervisedSubprocessExit:
    def test_propagates_the_exit_code_of_the_one_subprocess_that_exits(self, launch_supervisor):
        """The exit code of the rank that dies is what the container exits with."""
        run = launch_supervisor(
            num_subprocesses=3,
            child_args=[
                "--mode",
                "exit",
                "--only-index",
                "1",
                "--exit-code",
                "7",
                "--record-signal",
                "--wait-for-reports",
                "3",
            ],
        )
        reports = run.wait_for_children(3)

        assert run.wait_for_exit() == 7
        assert _modes(reports) == {"0": "sleep", "1": "exit", "2": "sleep"}
        wait_until(
            lambda: {path.name for path in run.report_dir.glob("signal-*")} >= {"signal-0", "signal-2"},
            message="the survivors to record the forwarded SIGTERM",
        )
        assert int((run.report_dir / "signal-0").read_text()) == signal.SIGTERM
        assert int((run.report_dir / "signal-2").read_text()) == signal.SIGTERM
        for index, report in reports.items():
            wait_until(lambda pid=report["pid"]: not is_alive(pid), message=f"subprocess {index} to be killed")

    def test_a_successful_exit_leaves_the_others_running(self, launch_supervisor):
        """A rank finishing cleanly is not a failure, so the rest of the batch keeps running without it."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "exit", "--only-index", "0", "--exit-code", "0", "--wait-for-reports", "2"],
        )
        reports = run.wait_for_children(2)
        wait_until(lambda: not is_alive(reports["0"]["pid"]), message="the exiting subprocess to be gone")

        time.sleep(2.5)

        assert run.process.poll() is None
        assert is_alive(reports["1"]["pid"])

    def test_a_subprocess_killed_by_a_signal_maps_to_128_plus_the_signal(self, launch_supervisor):
        """A subprocess dying from SIGKILL surfaces as exit code 137, the shell convention."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "suicide", "--only-index", "0", "--wait-for-reports", "2"],
        )
        reports = run.wait_for_children(2)

        assert run.wait_for_exit() == 128 + signal.SIGKILL
        assert _modes(reports) == {"0": "suicide", "1": "sleep"}


class TestSignalHandling:
    def test_sigterm_is_forwarded_to_every_subprocess(self, launch_supervisor):
        """SIGTERM sent to the supervisor reaches every subprocess rather than stopping at it."""
        run = launch_supervisor(num_subprocesses=2, child_args=["--mode", "record-signal"])
        run.wait_for_children(2)

        os.kill(run.process.pid, signal.SIGTERM)

        assert run.wait_for_exit() == 128 + signal.SIGTERM
        assert _recorded_signals(run) == {"0": signal.SIGTERM, "1": signal.SIGTERM}

    def test_sigint_is_forwarded_to_every_subprocess(self, launch_supervisor):
        """SIGINT is forwarded as SIGINT rather than translated to SIGTERM, and yields exit code 130."""
        run = launch_supervisor(num_subprocesses=2, child_args=["--mode", "record-signal"])
        run.wait_for_children(2)

        os.kill(run.process.pid, signal.SIGINT)

        assert run.wait_for_exit() == 128 + signal.SIGINT
        assert _recorded_signals(run) == {"0": signal.SIGINT, "1": signal.SIGINT}

    def test_grandchildren_of_a_running_subprocess_die_with_its_group(self, launch_supervisor):
        """Forwarding is per process group, so a process spawned by a subprocess is taken down too."""
        run = launch_supervisor(num_subprocesses=1, child_args=["--mode", "sleep", "--spawn-grandchild"])
        reports = run.wait_for_children(1)
        grandchild_pid = reports["0"]["grandchild_pid"]

        os.kill(run.process.pid, signal.SIGTERM)

        assert run.wait_for_exit() == 128 + signal.SIGTERM
        wait_until(lambda: not is_alive(grandchild_pid), message="the grandchild to be killed")

    def test_a_subprocess_ignoring_sigterm_is_killed_after_the_grace_period(self, launch_supervisor):
        """A subprocess that ignores SIGTERM is SIGKILLed once the grace period expires."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "ignore-sigterm"],
            termination_grace_period_seconds=GRACE_PERIOD_SECONDS,
        )
        reports = run.wait_for_children(2)
        started_at = time.monotonic()

        os.kill(run.process.pid, signal.SIGTERM)

        assert run.wait_for_exit() == 128 + signal.SIGTERM
        elapsed = time.monotonic() - started_at
        assert GRACE_PERIOD_SECONDS <= elapsed <= GRACE_PERIOD_SECONDS + GRACE_PERIOD_UPPER_BOUND_SLACK_SECONDS
        for index, report in reports.items():
            wait_until(lambda pid=report["pid"]: not is_alive(pid), message=f"subprocess {index} to be killed")

    def test_an_unreadable_torchelastic_error_file_fails_the_pod_instead_of_hanging_it(self, launch_supervisor):
        """torchelastic reads that file while bookkeeping a failure, before it stops the tee threads."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=[
                "--mode",
                "exit",
                "--only-index",
                "0",
                "--exit-code",
                "1",
                "--corrupt-error-file",
                "--wait-for-reports",
                "2",
            ],
        )
        reports = run.wait_for_children(2)

        assert run.wait_for_exit() != 0
        for index, report in reports.items():
            wait_until(lambda pid=report["pid"]: not is_alive(pid), message=f"subprocess {index} to be gone")

    def test_an_inherited_signal_list_cannot_disable_the_forwarding(self, launch_supervisor):
        """A stale value for torchelastic's own variable would kill the supervisor without forwarding."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "record-signal"],
            extra_env={_SIGNALS_TO_HANDLE_ENV_VAR: "SIGUSR1"},
        )
        run.wait_for_children(2)

        os.kill(run.process.pid, signal.SIGTERM)

        assert run.wait_for_exit() == 128 + signal.SIGTERM
        assert _recorded_signals(run) == {"0": signal.SIGTERM, "1": signal.SIGTERM}

    def test_a_second_signal_during_the_grace_period_does_not_abort_the_teardown(self, launch_supervisor):
        """Two SIGTERMs are ordinary, and the second must not leave the batch half torn down."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "ignore-sigterm"],
            termination_grace_period_seconds=DOUBLE_SIGNAL_GRACE_PERIOD_SECONDS,
            capture_output=True,
        )
        reports = run.wait_for_children(2)

        os.kill(run.process.pid, signal.SIGTERM)
        time.sleep(SECOND_SIGNAL_DELAY_SECONDS)
        os.kill(run.process.pid, signal.SIGTERM)
        _, stderr = run.process.communicate(timeout=SUPERVISOR_EXIT_TIMEOUT_SECONDS)

        assert run.process.returncode == 128 + signal.SIGTERM
        assert "Received SIGTERM while already tearing down" in stderr
        for index, report in reports.items():
            wait_until(lambda pid=report["pid"]: not is_alive(pid), message=f"subprocess {index} to be killed")


@pytest.fixture
def launch_supervisor(tmp_path: Path) -> Iterator[Callable[..., SupervisorRun]]:
    runs: list[SupervisorRun] = []

    def launch(
        *,
        num_subprocesses: int,
        child_args: list[str],
        termination_grace_period_seconds: float = 5.0,
        command: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        capture_output: bool = False,
    ) -> SupervisorRun:
        supervised = command or [sys.executable, str(CHILD_SCRIPT), "--report-dir", str(tmp_path), *child_args]
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "miles.utils.workers.process_supervisor",
                "--num-subprocesses",
                str(num_subprocesses),
                "--termination-grace-period-seconds",
                str(termination_grace_period_seconds),
                "--",
                *supervised,
            ],
            cwd=REPO_ROOT,
            env={**_repo_environment(), **(extra_env or {})},
            start_new_session=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
        )
        run = SupervisorRun(process=process, report_dir=tmp_path)
        runs.append(run)
        return run

    yield launch

    for run in runs:
        signal_process_group(run.process.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            run.process.wait(timeout=SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS)

        signal_process_group(run.process.pid, signal.SIGKILL)
        for report in run.reports().values():
            signal_process_group(report["pid"], signal.SIGKILL)
            if "grandchild_pid" in report:
                kill_process(report["grandchild_pid"])
        run.process.wait(timeout=SUPERVISOR_EXIT_TIMEOUT_SECONDS)
        for stream in (run.process.stdout, run.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


@pytest.fixture(autouse=True)
def restored_signal_handlers() -> Iterator[None]:
    saved = {forwarded: signal.getsignal(forwarded) for forwarded in (signal.SIGTERM, signal.SIGINT)}

    yield

    for forwarded, handler in saved.items():
        signal.signal(forwarded, handler)


@pytest.fixture
def unconfigured_root_logging(monkeypatch) -> None:
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)


@dataclass
class SupervisorRun:
    process: subprocess.Popen
    report_dir: Path

    def wait_for_children(self, count: int) -> dict[str, dict]:
        wait_until(lambda: len(self.reports()) == count, message=f"{count} children to report in")
        return self.reports()

    def reports(self) -> dict[str, dict]:
        reports = {}
        for path in sorted(self.report_dir.glob("child-*.json")):
            reports[path.stem.removeprefix("child-")] = json.loads(path.read_text())
        return reports

    def wait_for_exit(self) -> int:
        return self.process.wait(timeout=SUPERVISOR_EXIT_TIMEOUT_SECONDS)


def _repo_environment() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]),
    }


def wait_until(predicate: Callable[[], bool], *, message: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)

    raise AssertionError(f"Timed out after {timeout}s waiting for {message}")


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not is_zombie(pid)


def is_zombie(pid: int) -> bool:
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.parent.parent.is_dir():
        try:
            return "State:\tZ" in proc_status.read_text()
        except FileNotFoundError:
            return False

    listing = subprocess.run(["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True)
    return listing.stdout.strip().startswith("Z")


def signal_process_group(pid: int, sent_signal: signal.Signals) -> None:
    try:
        os.killpg(pid, sent_signal)
    except (ProcessLookupError, PermissionError):
        pass


def kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _fake_context(
    *,
    num_subprocesses: int = 1,
    wait_outcome: RunProcsResult | BaseException | None = None,
) -> _FakeContext:
    pids = {index: CHILD_PID + index for index in range(num_subprocesses)}
    return _FakeContext(pids_by_index=pids, wait_outcome=wait_outcome)


def _run_result(failures: dict[int, ProcessFailure] | None = None) -> RunProcsResult:
    return RunProcsResult(failures=failures or {})


def _failure(*, local_rank: int = 0, exitcode: int, timestamp: int = 0) -> ProcessFailure:
    failure = ProcessFailure(
        local_rank=local_rank,
        pid=CHILD_PID + local_rank,
        exitcode=exitcode,
        error_file=os.devnull,
    )
    failure.timestamp = timestamp
    return failure


def _record_supervisor(monkeypatch) -> list[dict]:
    created: list[dict] = []

    def supervise(**kwargs) -> int:
        created.append(kwargs)
        return RECORDED_SUPERVISOR_EXIT_CODE

    monkeypatch.setattr(supervisor_module, "supervise", supervise)
    return created


def _modes(reports: dict[str, dict]) -> dict[str, str]:
    return {index: report["mode"] for index, report in reports.items()}


def _recorded_signals(run: SupervisorRun) -> dict[str, int]:
    expected = len(run.reports())
    wait_until(
        lambda: len(list(run.report_dir.glob("signal-*"))) == expected,
        message="every subprocess to record the signal it received",
    )
    return {path.name.removeprefix("signal-"): int(path.read_text()) for path in run.report_dir.glob("signal-*")}


@dataclass
class _FakeContext:
    pids_by_index: dict[int, int]
    wait_outcome: RunProcsResult | BaseException | None = None
    close_calls: list[tuple[signal.Signals | None, float]] = field(default_factory=list)
    handlers_during_close: list[tuple[object, object]] = field(default_factory=list)

    def pids(self) -> dict[int, int]:
        return self.pids_by_index

    def wait(self) -> RunProcsResult | None:
        if isinstance(self.wait_outcome, BaseException):
            raise self.wait_outcome
        return self.wait_outcome

    def close(self, death_sig: signal.Signals | None = None, timeout: float = 30) -> None:
        self.close_calls.append((death_sig, timeout))
        self.handlers_during_close.append((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)))


def _child_main() -> int:
    args = _build_parser().parse_args()
    report_dir = Path(args.report_dir)
    index = os.environ.get(SUBPROCESS_INDEX_ENV_VAR, "missing")
    selected = args.only_index is None or args.only_index == index
    mode = args.mode if selected else MODE_SLEEP

    report: dict[str, object] = {
        "index": index,
        "pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "session_id": os.getsid(0),
        "mode": mode,
        "env_marker": os.environ.get(ENV_MARKER_VAR),
    }

    if args.corrupt_error_file:
        Path(os.environ["TORCHELASTIC_ERROR_FILE"]).write_text('{"message": ')

    if args.emit_stream_markers:
        print(f"{STDOUT_MARKER}{index}", flush=True)
        print(f"{STDERR_MARKER}{index}", file=sys.stderr, flush=True)

    if mode == MODE_IGNORE_SIGTERM:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == MODE_RECORD_SIGNAL or args.record_signal:
        recorder = _make_signal_recorder(report_dir=report_dir, index=index)
        signal.signal(signal.SIGTERM, recorder)
        signal.signal(signal.SIGINT, recorder)
    if args.spawn_grandchild and selected:
        report["grandchild_pid"] = _spawn_grandchild(report_dir=report_dir, index=index)

    _write_report(report_dir=report_dir, index=index, report=report)
    if selected and args.wait_for_reports is not None:
        _wait_for_reports(report_dir=report_dir, count=args.wait_for_reports)

    if mode == MODE_EXIT:
        _publish_atomically(path=report_dir / f"exited-at-{index}", content=repr(time.monotonic()))
        return args.exit_code
    if mode == MODE_SUICIDE:
        os.kill(os.getpid(), signal.SIGKILL)

    _sleep_forever()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[MODE_SLEEP, MODE_EXIT, MODE_SUICIDE, MODE_IGNORE_SIGTERM, MODE_RECORD_SIGNAL],
    )
    parser.add_argument("--only-index", default=None)
    parser.add_argument("--record-signal", action="store_true")
    parser.add_argument("--spawn-grandchild", action="store_true")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--wait-for-reports", type=int, default=None)
    parser.add_argument("--emit-stream-markers", action="store_true")
    parser.add_argument("--corrupt-error-file", action="store_true")
    return parser


def _wait_for_reports(*, report_dir: Path, count: int) -> None:
    _wait_until(
        lambda: len(list(report_dir.glob("child-*.json"))) >= count,
        message=f"{count} sibling reports",
    )


def _spawn_grandchild(*, report_dir: Path, index: str) -> int:
    ready_path = report_dir / f"grandchild-ready-{index}"
    grandchild = subprocess.Popen([sys.executable, "-c", _GRANDCHILD, str(ready_path)])
    _wait_until(ready_path.exists, message="the grandchild to report ready")
    return grandchild.pid


def _wait_until(predicate: Callable[[], bool], *, message: str) -> None:
    deadline = time.monotonic() + REPORT_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)

    raise TimeoutError(f"Timed out waiting for {message}")


def _make_signal_recorder(*, report_dir: Path, index: str) -> Callable[[int, FrameType | None], None]:
    def handler(signum: int, frame: FrameType | None) -> None:
        _publish_atomically(path=report_dir / f"signal-{index}", content=str(signum))
        sys.exit(0)

    return handler


def _write_report(*, report_dir: Path, index: str, report: dict[str, object]) -> None:
    _publish_atomically(path=report_dir / f"child-{index}.json", content=json.dumps(report))


def _publish_atomically(*, path: Path, content: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(content)
    os.replace(temp_path, path)


def _sleep_forever() -> None:
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(_child_main())
