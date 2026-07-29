import logging
import os
import shutil
import subprocess
import threading
import time

import psutil

logger = logging.getLogger(__name__)

DUMP_INTERVAL_ENV = "MILES_DEBUG_PYSPY_DUMP_INTERVAL"
_PROCESS_KEYWORDS = ("python", "ray::", "sglang::")
_PER_PROCESS_DUMP_TIMEOUT_SECONDS = 60

_started = False


def maybe_start_periodic_pyspy_dump() -> None:
    global _started

    interval = float(os.environ.get(DUMP_INTERVAL_ENV, "0") or "0")
    if interval <= 0 or _started:
        return

    if shutil.which("py-spy") is None:
        logger.error(f"{DUMP_INTERVAL_ENV}={interval} set but py-spy not found on PATH; skipping periodic dump")
        return

    _started = True
    thread = threading.Thread(
        target=_dump_loop,
        args=(interval,),
        daemon=True,
        name="debug-pyspy-dump",
    )
    thread.start()
    logger.info(f"Started periodic py-spy dump (interval={interval}s)")


def _dump_loop(interval: float) -> None:
    while True:
        time.sleep(interval)
        try:
            _dump_all_processes()
        except Exception:
            logger.exception("Periodic py-spy dump iteration failed")


def _dump_all_processes() -> None:
    targets = _collect_target_processes()
    print(
        f"===== [debug-pyspy] ts={int(time.time())} processes={len(targets)} =====",
        flush=True,
    )
    for pid, cmdline in targets:
        _dump_one_process(pid=pid, cmdline=cmdline)


def _collect_target_processes() -> list[tuple[int, str]]:
    targets: list[tuple[int, str]] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except (psutil.Error, OSError):
            continue
        if any(keyword in cmdline for keyword in _PROCESS_KEYWORDS):
            targets.append((proc.info["pid"], cmdline[:160]))
    return targets


def _dump_one_process(pid: int, cmdline: str) -> None:
    for native_flag in ("--native", ""):
        cmd = f"py-spy dump {native_flag} --pid {pid}".strip()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_PER_PROCESS_DUMP_TIMEOUT_SECONDS,
                check=True,
            )
            print(f"----- [debug-pyspy] pid={pid} cmd={cmdline}\n{result.stdout}", flush=True)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr = (getattr(e, "stderr", "") or "").strip()
            if native_flag == "":
                print(f"----- [debug-pyspy] pid={pid} cmd={cmdline} DUMP FAILED: {stderr[:200]}", flush=True)
