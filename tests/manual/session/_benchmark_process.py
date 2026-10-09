"""Run Linux CPU benchmarks with a deadline and an owned-process RSS budget."""

import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time

import psutil


def run_benchmark_process(command, *, cwd, env, timeout=1200, memory_bytes=12_000_000_000):
    """Run the guard in isolation so it owns even orphaned benchmark descendants."""
    policy = {"command": command, "cwd": str(cwd), "timeout": timeout, "memory_bytes": memory_bytes}
    result = subprocess.run(
        [sys.executable, __file__, json.dumps(policy)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _supervise(command, cwd, timeout, memory_bytes):
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot adopt benchmark descendants")
    owner = psutil.Process()
    handles = {}
    failure = None
    deadline = time.monotonic() + timeout
    peak_rss = 0

    def alive(fd):
        return not select.select([fd], [], [], 0)[0]

    def discover():
        for process in owner.children(recursive=True):
            try:
                identity = (process.pid, process.create_time())
                if identity in handles:
                    continue
                fd = os.pidfd_open(process.pid)
                try:
                    if psutil.Process(process.pid).create_time() != identity[1]:
                        raise RuntimeError("Benchmark process identity changed")
                except BaseException:
                    os.close(fd)
                    raise
                handles[identity] = fd
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue

    with tempfile.TemporaryFile(mode="w+") as log:
        child = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                discover()
                rss = 0
                for (pid, created), fd in handles.items():
                    if alive(fd):
                        try:
                            process = psutil.Process(pid)
                            if process.create_time() == created:
                                rss += process.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                peak_rss = max(peak_rss, rss)
                if rss > memory_bytes:
                    failure = f"Benchmark RSS {rss} exceeds budget {memory_bytes}"
                    break
                if time.monotonic() >= deadline:
                    failure = f"Benchmark exceeded {timeout}s deadline"
                    break
                time.sleep(0.1)
        finally:
            if child.poll() is not None:
                until = time.monotonic() + 2
                while time.monotonic() < until:
                    discover()
                    if not any(alive(fd) for fd in handles.values()):
                        break
                    time.sleep(0.05)
            # Re-discover adopted children during cleanup; pidfds cannot target a reused PID.
            for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
                until = time.monotonic() + grace
                sent = set()
                while time.monotonic() < until:
                    discover()
                    live = [fd for fd in handles.values() if alive(fd)]
                    if not live:
                        break
                    if failure is None and child.poll() is not None:
                        failure = "Benchmark left child processes running"
                    for fd in live:
                        if fd not in sent:
                            try:
                                signal.pidfd_send_signal(fd, sig)
                            except ProcessLookupError:
                                pass
                            sent.add(fd)
                    time.sleep(0.1)
            residual = any(alive(fd) for fd in handles.values())
            for (pid, _), fd in handles.items():
                os.close(fd)
                if pid != child.pid:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            assert not residual, "Benchmark cleanup left live processes"
            child.wait(timeout=5)
        log.seek(0)
        output = log.read()
    assert failure is None and child.returncode == 0, f"{failure or 'Benchmark failed'}; peak RSS={peak_rss}\n{output}"
    return output


if __name__ == "__main__":
    print(_supervise(**json.loads(sys.argv[1])), end="")
