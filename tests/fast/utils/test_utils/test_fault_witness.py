import os
from pathlib import Path

import pytest

from miles.utils.test_utils import fault_witness
from miles.utils.test_utils.fault_witness import DeadlockTarget, deadlock_evidence


class TestDeadlockEvidence:
    @pytest.mark.parametrize(
        "changed",
        [None, "missing_held", "missing_wait", "other_pid", "other_inode", "read_lock", "other_lock_id", "duplicate"],
    )
    def test_only_the_matching_self_blocked_lock_is_evidence(self, changed: str | None) -> None:
        """A lock owned elsewhere or merely held without a waiter cannot prove self-deadlock."""
        target = DeadlockTarget(thread_id=43, device=os.makedev(0, 1), inode=99, holds_gil=True)
        held = "3: FLOCK ADVISORY WRITE 42 00:01:99 0 EOF"
        waiting = "3: -> FLOCK ADVISORY WRITE 42 00:01:99 0 EOF"
        if changed == "other_pid":
            waiting = waiting.replace("WRITE 42", "WRITE 44")
        elif changed == "other_inode":
            waiting = waiting.replace(":99", ":100")
        elif changed == "read_lock":
            waiting = waiting.replace("WRITE", "READ")
        elif changed == "other_lock_id":
            waiting = waiting.replace("3:", "4:", 1)
        lines = ([] if changed == "missing_held" else [held]) + ([] if changed == "missing_wait" else [waiting])
        if changed == "duplicate":
            lines.append(waiting)
        evidence = deadlock_evidence("\n".join(lines), pid=42, target=target)
        assert evidence == ([held, waiting] if changed is None else None)


class TestStopWitness:
    @pytest.mark.parametrize("states", [("T", "T"), ("T", "S"), ("t", "t"), ()])
    def test_only_a_nonempty_fully_stopped_process_is_confirmed(
        self, states: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running threads, debugger stops and missing tasks cannot prove SIGSTOP."""
        for index, state in enumerate(states):
            task = tmp_path / "42" / "task" / str(index)
            task.mkdir(parents=True)
            (task / "stat").write_text(f"{index} (a tricky) process) {state} 0 0 0\n")
        ticks = iter([0.0, 0.0, 11.0])
        monkeypatch.setattr(fault_witness, "Path", lambda _: tmp_path)
        monkeypatch.setattr(fault_witness.select, "select", lambda *args: ([], [], []))
        monkeypatch.setattr(fault_witness.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(fault_witness.time, "sleep", lambda _: None)

        if states == ("T", "T"):
            fault_witness.wait_process_stopped(pid=42, pidfd=99)
        else:
            with pytest.raises(TimeoutError):
                fault_witness.wait_process_stopped(pid=42, pidfd=99)

    def test_exit_during_stop_observation_is_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exiting target cannot turn a stale proc snapshot into stop evidence."""
        task = tmp_path / "42" / "task" / "42"
        task.mkdir(parents=True)
        (task / "stat").write_text("42 (target) T 0 0 0\n")
        checks = iter([([], [], []), ([99], [], [])])
        monkeypatch.setattr(fault_witness, "Path", lambda _: tmp_path)
        monkeypatch.setattr(fault_witness.select, "select", lambda *args: next(checks))

        with pytest.raises(ProcessLookupError, match="during stop observation"):
            fault_witness.wait_process_stopped(pid=42, pidfd=99)
