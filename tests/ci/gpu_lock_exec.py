#!/usr/bin/env python3
import argparse
import fcntl
import os
import sys
import time
from typing import List, Tuple

SLEEP_BACKOFF = 1.0


def main():
    args = _parse_args()

    if args.print_only:
        _execute_print_only(args)
        return

    locks = _try_acquire(args)

    dev_list = ",".join(str(d) for d, _ in locks)
    os.environ[args.target_env_name] = dev_list
    print(f"[gpu_lock_exec] Acquired GPUs: {dev_list}", flush=True)

    _os_execvp(args)


def _os_execvp(args):
    cmd = args.cmd
    if cmd[0] == "--":
        cmd = cmd[1:]
    os.execvp(cmd[0], cmd)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=None, help="Acquire this many GPUs (any free ones)")
    p.add_argument("--devices", type=str, default=None, help="Comma separated explicit devices to acquire (e.g. 0,1)")
    p.add_argument("--total-gpus", type=int, default=8, help="Total GPUs on the machine")
    p.add_argument("--timeout", type=int, default=3600, help="Seconds to wait for locks before failing")
    p.add_argument(
        "--target-env-name", type=str, default="CUDA_VISIBLE_DEVICES", help="Which env var to set for devices"
    )
    p.add_argument("--lock-dir", type=str, default="/dev/shm", help="Directory where lock files live")
    p.add_argument(
        "--lock-pattern",
        type=str,
        default="custom_gpu_lock_{i}.lock",
        help='Filename pattern with "{i}" placeholder, e.g. "custom_gpu_lock_{i}.lock"',
    )
    p.add_argument("--print-only", action="store_true", help="Probe free devices and print them (does NOT hold locks)")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to exec after '--' (required unless --print-only)")
    args = p.parse_args()

    if "{i}" not in args.lock_pattern:
        raise Exception("ERROR: --lock-pattern must contain '{i}' placeholder.")

    if not args.cmd:
        raise Exception("ERROR: missing command to run. Use -- before command.")

    return args


def _execute_print_only(args):
    free = []
    _ensure_lock_files(lock_dir=args.lock_dir, pattern=args.lock_pattern, total_gpus=args.total_gpus)
    for i in range(args.total_gpus):
        path = _get_lock_path(lock_dir=args.lock_dir, pattern=args.lock_pattern, i=i)
        try:
            fd = open(path, "w")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                free.append(i)
            except BlockingIOError:
                pass
            fd.close()
        except Exception as e:
            print(f"Warning: Error while probing lock {path}: {e}", file=sys.stderr)

    print("Free GPUs:", ",".join(str(x) for x in free))


def _try_acquire(args):
    if args.devices:
        devs = _parse_devices(args.devices)
        return _try_acquire_specific(devs, args.lock_dir, args.lock_pattern, args.timeout)
    else:
        return _try_acquire_count(args.count, args.total_gpus, args.lock_dir, args.lock_pattern, args.timeout)


def _try_acquire_specific(devs: List[int], lock_dir: str, pattern: str, timeout: int):
    fds = []
    start = time.time()
    try:
        _ensure_lock_files(lock_dir, pattern, max(devs) + 1)
        for d in devs:
            fd_lock = FdLock(lock_dir, pattern, gpu_id=dev)
            fd_lock.open()
            while True:
                try:
                    fd_lock.lock()
                    break
                except BlockingIOError:
                    if time.time() - start > timeout:
                        raise TimeoutError(f"Timeout while waiting for GPU {d}")
                    time.sleep(SLEEP_BACKOFF)
            fds.append(fd_lock)
        return fds
    except Exception as e:
        print(f"Error during specific GPU acquisition: {e}", file=sys.stderr)
        for fd_lock in fds:
            fd_lock.close()
        raise


def _try_acquire_count(count: int, total_gpus: int, lock_dir: str, pattern: str, timeout: int):
    start = time.time()
    _ensure_lock_files(lock_dir, pattern, total_gpus)
    while True:
        acquired: List = []
        for i in range(total_gpus):
            fd_lock = FdLock(lock_dir, pattern, gpu_id=i)
            try:
                fd_lock.open()
            except Exception as e:
                print(f"Warning: Could not open lock file: {e}", file=sys.stderr)
                continue
            try:
                fd_lock.lock()
                acquired.append(fd_lock)
                if len(acquired) == count:
                    return acquired
            except BlockingIOError:
                for fd_lock in acquired:
                    fd_lock.close()
                acquired = []
                break
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout acquiring {count} GPUs (out of {total_gpus})")
        time.sleep(SLEEP_BACKOFF)

class FdLock:
    def __init__(self, lock_dir, pattern, gpu_id: int):
        self.gpu_id = gpu_id
        self.path = _get_lock_path(lock_dir, pattern, self.gpu_id)
        self.fd = None

    def open(self):
        assert self.fd is None
        self.fd = open(self.path, "w")

    def lock(self):
        assert self.fd is not None
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self):
        assert self.fd is not None
        try:
            self.fd.close()
        except Exception as e:
            print(f"Warning: Failed to close file descriptor: {e}", file=sys.stderr)
        self.fd = None

def _ensure_lock_files(lock_dir: str, pattern: str, total_gpus: int):
    os.makedirs(lock_dir, exist_ok=True)
    for i in range(total_gpus):
        p = _get_lock_path(lock_dir, pattern, i)
        try:
            open(p, "a").close()
        except Exception as e:
            print(f"Warning: Could not create lock file {p}: {e}", file=sys.stderr)


def _get_lock_path(lock_dir: str, pattern: str, i: int) -> str:
    return os.path.join(lock_dir, pattern.format(i=i))


def _parse_devices(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


if __name__ == "__main__":
    main()
