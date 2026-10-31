"""Names for what a weight sync leaves on a host, so a later run can tell whether the process that wrote it is gone.

Ray kills a trainer and every process it started with a signal no handler sees, so nothing can clean up at exit.
"""

import os
import socket


def owner_tag() -> str:
    """This process as `<host>-<pid namespace>-<pid>`, which no other live process shares."""
    return f"{socket.gethostname()}-{_pid_namespace()}-{os.getpid()}"


def owner_is_gone(tag: str) -> bool:
    """Whether the process `tag` names has exited; only decidable on its host and in its pid namespace."""
    parts = tag.rsplit("-", 2)
    if len(parts) != 3:
        return False
    host, namespace, pid = parts
    if host != socket.gethostname() or namespace != _pid_namespace() or not pid.isdigit():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    return False


def _pid_namespace() -> str:
    return str(os.stat("/proc/self/ns/pid").st_ino)
