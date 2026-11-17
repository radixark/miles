import signal
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.fast.utils.soak.k8s_utils.pod_fakes import _FakeProcKernel
from tests.utils.soak.k8s_utils.pod_processes import (
    ProcessIdentity,
    ProcessSignal,
    ProcessSignalReceipt,
    ProcessTarget,
    observe_processes,
    signal_observed_processes,
)
from tests.utils.soak.k8s_utils.pod_processes_cli import app
from typer.testing import CliRunner

_PATTERN = "sglang::"


@pytest.fixture
def kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeProcKernel:
    kernel = _FakeProcKernel(tmp_path)
    kernel.install(monkeypatch)
    return kernel


def _observe_engines(kernel: _FakeProcKernel, *pids: int) -> ProcessTarget:
    for pid in pids:
        kernel.add(pid, cmdline=f"sglang::scheduler_{pid} --tp 1", start_ticks=100 + pid)
    return observe_processes(pod_uid=kernel.pod_uid, pattern=_PATTERN)


def _target(**overrides: object) -> ProcessTarget:
    return ProcessTarget(
        **{
            "pod_uid": "uid-pod-a",
            "boot_id": "boot-a",
            "pid_namespace": "pidns",
            "init_start_ticks": 11,
            "pattern": _PATTERN,
            "processes": [ProcessIdentity(pid=41, start_ticks=5), ProcessIdentity(pid=42, start_ticks=6)],
            **overrides,
        }
    )


def _receipt(**overrides: object) -> ProcessSignalReceipt:
    return ProcessSignalReceipt(
        **{"request_id": "req-1", "target": _target(), "operation": ProcessSignal.STOP, "signalled_pids": [41, 42]}
        | overrides
    )


def _last_line(stdout: str) -> str:
    return stdout.strip().splitlines()[-1]


class TestProcessSignalReceiptValidateFor:
    def test_the_exact_request_target_operation_and_pids_validate(self) -> None:
        """A receipt echoing the request, incarnation, operation and every pid is accepted."""
        _receipt().validate_for(request_id="req-1", target=_target(), operation=ProcessSignal.STOP)

    @pytest.mark.parametrize(
        "receipt",
        [
            pytest.param(_receipt(request_id="req-2"), id="other_request"),
            pytest.param(_receipt(target=_target(boot_id="boot-b")), id="other_boot"),
            pytest.param(
                _receipt(
                    target=_target(
                        processes=[ProcessIdentity(pid=41, start_ticks=5), ProcessIdentity(pid=42, start_ticks=9)]
                    )
                ),
                id="reused_pid",
            ),
            pytest.param(_receipt(operation=ProcessSignal.KILL), id="kill_for_stop"),
            pytest.param(_receipt(signalled_pids=[41]), id="incomplete"),
            pytest.param(_receipt(signalled_pids=[42, 41]), id="reordered"),
            pytest.param(_receipt(signalled_pids=[41, 42, 43]), id="extra_pid"),
        ],
    )
    def test_any_mismatch_rejects_the_receipt(self, receipt: ProcessSignalReceipt) -> None:
        """A receipt from another request, incarnation or operation, or covering other pids, proves nothing."""
        with pytest.raises(AssertionError):
            receipt.validate_for(request_id="req-1", target=_target(), operation=ProcessSignal.STOP)

    def test_a_receipt_needs_a_request_id_and_at_least_one_pid(self) -> None:
        """Empty request ids and empty pid lists cannot be serialized as a receipt."""
        with pytest.raises(ValidationError):
            _receipt(request_id="")
        with pytest.raises(ValidationError):
            _receipt(signalled_pids=[])


class TestObserveProcesses:
    def test_it_records_matching_pids_with_the_container_identity(self, kernel: _FakeProcKernel) -> None:
        """Only matching processes are kept, sorted, alongside the boot, namespace and init identity."""
        kernel.add(30, cmdline="python -m sglang::detokenizer", start_ticks=300)
        kernel.add(10, cmdline="python -m sglang::scheduler", start_ticks=100)
        kernel.add(20, cmdline="python train.py", start_ticks=200)

        target = observe_processes(pod_uid="uid-pod-a", pattern=_PATTERN)

        assert target == ProcessTarget(
            pod_uid="uid-pod-a",
            boot_id="boot-a",
            pid_namespace="pid:[4026531836]",
            init_start_ticks=11,
            pattern=_PATTERN,
            processes=[ProcessIdentity(pid=10, start_ticks=100), ProcessIdentity(pid=30, start_ticks=300)],
        )

    def test_init_itself_and_its_parent_are_never_targets(self, kernel: _FakeProcKernel) -> None:
        """The CLI's own process tree matches the pattern argument but must not be signalled."""
        kernel.add(2, cmdline="python -m pod_processes_cli observe uid sglang::")
        kernel.add(3, cmdline="sh -c kubectl exec sglang::")
        kernel.processes[1].cmdline = "sglang::init"
        kernel.write(kernel.processes[1])
        kernel.add(50, cmdline="sglang::scheduler")

        assert [process.pid for process in observe_processes(pod_uid="uid-pod-a", pattern=_PATTERN).processes] == [50]

    def test_a_process_that_vanishes_during_the_scan_is_skipped(self, kernel: _FakeProcKernel) -> None:
        """A /proc entry disappearing mid-scan is skipped rather than failing the observation."""
        kernel.add(50, cmdline="sglang::scheduler")
        (kernel.root / "proc" / "60").mkdir()

        assert [process.pid for process in observe_processes(pod_uid="uid-pod-a", pattern=_PATTERN).processes] == [50]

    def test_a_different_pod_uid_is_refused(self, kernel: _FakeProcKernel) -> None:
        """Observing from inside another pod incarnation would target the wrong processes."""
        kernel.add(50, cmdline="sglang::scheduler")

        with pytest.raises(AssertionError, match="Pod identity changed"):
            observe_processes(pod_uid="uid-pod-b", pattern=_PATTERN)

    def test_no_matching_process_is_not_a_target(self, kernel: _FakeProcKernel) -> None:
        """A pod with no engine process yields no target at all instead of an empty one."""
        kernel.add(20, cmdline="python train.py")

        with pytest.raises(ValidationError):
            observe_processes(pod_uid="uid-pod-a", pattern=_PATTERN)


class TestSignalObservedProcessesKill:
    def test_every_pidfd_is_opened_before_the_first_kill_and_all_are_closed(self, kernel: _FakeProcKernel) -> None:
        """All identities are pinned before any signal so a partial kill cannot race a pid reuse."""
        target = _observe_engines(kernel, 40, 41)

        assert signal_observed_processes(target=target, operation=ProcessSignal.KILL) == [40, 41]

        assert [(action, pid) for action, pid, _ in kernel.journal] == [
            ("open", 40),
            ("open", 41),
            ("signal", 40),
            ("signal", 41),
            ("close", 41),
            ("close", 40),
        ]
        assert kernel.signals() == [(40, "SIGKILL"), (41, "SIGKILL")]
        assert kernel.open_fds == set()

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(lambda kernel: setattr(kernel, "pod_uid", "uid-pod-b"), id="pod_uid"),
            pytest.param(lambda kernel: kernel.set_boot_id("boot-b"), id="boot_id"),
            pytest.param(lambda kernel: setattr(kernel, "pid_namespace", "pid:[1]"), id="pid_namespace"),
            pytest.param(lambda kernel: kernel.restart_init(99), id="container_restart"),
        ],
    )
    def test_a_changed_container_identity_sends_nothing(
        self,
        kernel: _FakeProcKernel,
        monkeypatch: pytest.MonkeyPatch,
        change: Callable[[_FakeProcKernel], object],
    ) -> None:
        """A rebooted host, new pod, new pid namespace or restarted container makes the observed pids stale."""
        target = _observe_engines(kernel, 40)
        change(kernel)
        kernel.install(monkeypatch)

        with pytest.raises(AssertionError):
            signal_observed_processes(target=target, operation=ProcessSignal.KILL)

        assert kernel.journal == []

    def test_a_reused_pid_is_refused_before_any_signal(self, kernel: _FakeProcKernel) -> None:
        """A pid now held by a process with other start ticks is not the observed engine."""
        target = _observe_engines(kernel, 40, 41)
        kernel.processes[41].start_ticks = 999
        kernel.write(kernel.processes[41])

        with pytest.raises(AssertionError, match="Process identity changed"):
            signal_observed_processes(target=target, operation=ProcessSignal.KILL)

        assert kernel.signals() == []
        assert kernel.open_fds == set()

    def test_a_changed_command_is_refused_before_any_signal(self, kernel: _FakeProcKernel) -> None:
        """A process that exec'd into something else no longer matches the observed pattern."""
        target = _observe_engines(kernel, 40)
        kernel.processes[40].cmdline = "python train.py"
        kernel.write(kernel.processes[40])

        with pytest.raises(AssertionError, match="Process command changed"):
            signal_observed_processes(target=target, operation=ProcessSignal.KILL)

        assert kernel.signals() == []

    def test_a_process_that_exited_before_injection_sends_nothing(self, kernel: _FakeProcKernel) -> None:
        """If any observed process already died the crash was not caused by this request."""
        target = _observe_engines(kernel, 40, 41)
        kernel.processes[41].exited = True

        with pytest.raises(ProcessLookupError, match="before injection"):
            signal_observed_processes(target=target, operation=ProcessSignal.KILL)

        assert kernel.signals() == []
        assert kernel.open_fds == set()

    def test_a_process_that_survives_the_kill_times_out(self, kernel: _FakeProcKernel) -> None:
        """Without every pidfd becoming readable the kill is not confirmed and no pids are returned."""
        target = _observe_engines(kernel, 40, 41)
        kernel.processes[41].survives_kill = True

        with pytest.raises(TimeoutError):
            signal_observed_processes(target=target, operation=ProcessSignal.KILL)

        assert kernel.open_fds == set()


class TestSignalObservedProcessesStop:
    def test_every_thread_stopped_confirms_the_stop_without_resuming(self, kernel: _FakeProcKernel) -> None:
        """A successful freeze leaves every thread in T and never sends SIGCONT."""
        target = _observe_engines(kernel, 40, 41)

        assert signal_observed_processes(target=target, operation=ProcessSignal.STOP) == [40, 41]

        assert kernel.signals() == [(40, "SIGSTOP"), (41, "SIGSTOP")]
        assert all(state == "T" for pid in (40, 41) for state in kernel.processes[pid].thread_states.values())
        assert kernel.open_fds == set()

    def test_an_already_stopped_process_is_refused_before_any_signal(self, kernel: _FakeProcKernel) -> None:
        """A process that was already frozen would make the stop look caused by this request."""
        target = _observe_engines(kernel, 40, 41)
        kernel.processes[41].thread_states[41] = "T"
        kernel.write(kernel.processes[41])

        with pytest.raises(ProcessLookupError, match="already stopped"):
            signal_observed_processes(target=target, operation=ProcessSignal.STOP)

        assert kernel.signals() == []

    def test_one_thread_that_never_stops_times_out_and_resumes_every_process(self, kernel: _FakeProcKernel) -> None:
        """A partially frozen process is not a witnessed stop; everything signalled is resumed."""
        target = _observe_engines(kernel, 40, 41)
        kernel.processes[41].unstoppable_tids = frozenset({42})

        with pytest.raises(TimeoutError):
            signal_observed_processes(target=target, operation=ProcessSignal.STOP)

        assert sorted(kernel.signals()) == [(40, "SIGCONT"), (40, "SIGSTOP"), (41, "SIGCONT"), (41, "SIGSTOP")]
        assert all(state == "S" for pid in (40, 41) for state in kernel.processes[pid].thread_states.values())
        assert kernel.clock == pytest.approx(5.0, abs=0.05)
        assert kernel.open_fds == set()

    def test_a_failed_send_resumes_only_the_processes_already_stopped(self, kernel: _FakeProcKernel) -> None:
        """A partial failure rolls back the earlier SIGSTOPs and never signals later processes."""
        target = _observe_engines(kernel, 40, 41, 43)
        kernel.processes[41].send_errors[signal.SIGSTOP] = PermissionError("denied")

        with pytest.raises(PermissionError):
            signal_observed_processes(target=target, operation=ProcessSignal.STOP)

        assert kernel.signals() == [(40, "SIGSTOP"), (40, "SIGCONT")]
        assert kernel.open_fds == set()

    def test_a_process_exiting_on_stop_is_refused_and_the_rest_resumed(self, kernel: _FakeProcKernel) -> None:
        """An exit during stop confirmation is a crash, not a freeze, and survivors are resumed."""
        target = _observe_engines(kernel, 40, 41)
        kernel.processes[41].exits_on_stop = True

        with pytest.raises(ProcessLookupError):
            signal_observed_processes(target=target, operation=ProcessSignal.STOP)

        assert (40, "SIGCONT") in kernel.signals()
        assert kernel.processes[40].thread_states == {40: "S", 41: "S"}

    def test_rollback_tolerates_a_process_that_already_exited(self, kernel: _FakeProcKernel) -> None:
        """Resuming an exited process is idempotent so the original failure still surfaces."""
        target = _observe_engines(kernel, 40, 41)
        kernel.processes[40].exits_on_stop = True

        with pytest.raises(ProcessLookupError, match="exited before stop was witnessed"):
            signal_observed_processes(target=target, operation=ProcessSignal.STOP)

        assert (41, "SIGCONT") in kernel.signals()


class TestPodProcessesCli:
    def test_stop_prints_a_receipt_that_validates_for_the_request(self, kernel: _FakeProcKernel) -> None:
        """The CLI reads the target from stdin and prints an exact receipt for this request and operation."""
        target = _observe_engines(kernel, 40, 41)

        result = CliRunner().invoke(app, ["stop", "req-9"], input=target.model_dump_json())

        assert result.exit_code == 0, result.output
        receipt = ProcessSignalReceipt.model_validate_json(_last_line(result.stdout))
        receipt.validate_for(request_id="req-9", target=target, operation=ProcessSignal.STOP)
        assert kernel.signals() == [(40, "SIGSTOP"), (41, "SIGSTOP")]

    def test_kill_prints_a_kill_receipt(self, kernel: _FakeProcKernel) -> None:
        """The kill command signals SIGKILL and describes the operation as KILL."""
        target = _observe_engines(kernel, 40)

        result = CliRunner().invoke(app, ["kill", "req-9"], input=target.model_dump_json())

        assert result.exit_code == 0, result.output
        assert ProcessSignalReceipt.model_validate_json(_last_line(result.stdout)).operation == ProcessSignal.KILL
        assert kernel.signals() == [(40, "SIGKILL")]

    def test_a_failed_signal_prints_no_receipt_and_exits_nonzero(self, kernel: _FakeProcKernel) -> None:
        """A refused injection must not print anything the form could parse as success."""
        target = _observe_engines(kernel, 40)
        kernel.processes[40].exited = True

        result = CliRunner().invoke(app, ["kill", "req-9"], input=target.model_dump_json())

        assert result.exit_code != 0
        assert "process_signal" not in result.stdout

    def test_observe_prints_the_observed_target(self, kernel: _FakeProcKernel) -> None:
        """The observe command emits the target JSON that the observer parses."""
        kernel.add(40, cmdline="sglang::scheduler", start_ticks=140)

        result = CliRunner().invoke(app, ["observe", "uid-pod-a", _PATTERN])

        assert result.exit_code == 0, result.output
        assert ProcessTarget.model_validate_json(_last_line(result.stdout)).processes == [
            ProcessIdentity(pid=40, start_ticks=140)
        ]
