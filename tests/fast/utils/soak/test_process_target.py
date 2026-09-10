from pathlib import Path

import pytest
from tests.utils.soak import process_target
from tests.utils.soak.process_target import ProcessIdentity, ProcessTarget

from miles.utils.workers.env_vars import POD_UID_ENV_VAR


@pytest.mark.parametrize("changed", [None, "pod", "boot", "namespace", "init", "process", "command"])
def test_replaced_identity_is_rejected_before_any_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str | None
) -> None:
    """Every observed identity must still match before any process receives a signal."""
    target = ProcessTarget(
        pod_uid="pod",
        boot_id="boot",
        pid_namespace="namespace",
        init_start_ticks=10,
        pattern="sglang::",
        processes=[ProcessIdentity(pid=42, start_ticks=20), ProcessIdentity(pid=43, start_ticks=30)],
    )
    boot_file = tmp_path / "sys/kernel/random/boot_id"
    boot_file.parent.mkdir(parents=True)
    boot_file.write_text("new" if changed == "boot" else "boot")
    for pid, ticks in [(1, 10), (42, 20), (43, 30)]:
        path = tmp_path / str(pid)
        path.mkdir()
        if (pid == 1 and changed == "init") or (pid == 43 and changed == "process"):
            ticks += 1
        (path / "stat").write_text(f"{pid} (a process (name)) " + " ".join(["S"] + ["0"] * 18 + [str(ticks)]))
        (path / "cmdline").write_bytes(b"replacement" if pid == 43 and changed == "command" else b"sglang::worker\0")

    monkeypatch.setattr(process_target, "Path", lambda path: tmp_path / Path(path).relative_to("/proc"))
    monkeypatch.setenv(POD_UID_ENV_VAR, "new" if changed == "pod" else "pod")
    monkeypatch.setattr(process_target.os, "readlink", lambda path: "new" if changed == "namespace" else "namespace")
    monkeypatch.setattr(process_target.os, "pidfd_open", lambda pid: pid + 1000, raising=False)
    closed: list[int] = []
    sent: list[int] = []
    monkeypatch.setattr(process_target.os, "close", closed.append)
    monkeypatch.setattr(process_target.signal, "pidfd_send_signal", lambda fd, sig: sent.append(fd), raising=False)

    if changed is None:
        assert process_target.kill_observed_processes(target) == [42, 43]
        assert sent == [1042, 1043]
        assert sorted(closed) == [1042, 1043]
    else:
        with pytest.raises(AssertionError):
            process_target.kill_observed_processes(target)
        assert sent == []
        if changed in {"process", "command"}:
            assert sorted(closed) == [1042, 1043]
