import os
import socket
import subprocess
import sys

from miles.backends.training_utils.weight_update.protocols.utils.owner import owner_is_gone, owner_tag


def _exited_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_a_live_process_is_never_gone():
    assert not owner_is_gone(owner_tag())


def test_an_exited_process_on_this_host_is_gone():
    host, namespace, _ = owner_tag().rsplit("-", 2)
    assert owner_is_gone(f"{host}-{namespace}-{_exited_pid()}")


def test_an_owner_elsewhere_is_never_taken_for_gone():
    """Another host or pid namespace may reuse the number for a live process, so its files stay."""
    host, namespace, _ = owner_tag().rsplit("-", 2)
    pid = _exited_pid()
    assert not owner_is_gone(f"other-host-{namespace}-{pid}")
    assert not owner_is_gone(f"{host}-1-{pid}")
    assert not owner_is_gone(f"{host}-{namespace}-not-a-pid")
    assert not owner_is_gone("hand-made")


def test_the_tag_names_this_process():
    assert owner_tag().startswith(f"{socket.gethostname()}-") and owner_tag().endswith(f"-{os.getpid()}")
