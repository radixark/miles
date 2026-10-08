import fcntl
import os
import select
import subprocess
from pathlib import Path


def find_worker_pid():
    pid = os.getppid()
    while pid > 1:
        proc = Path(f"/proc/{pid}")
        if (proc / "comm").read_text().strip() == "Runner.Worker":
            return pid
        status = (proc / "status").read_text().splitlines()
        pid = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
    raise RuntimeError("B200 job hook must run beneath Runner.Worker")


def acquire(worker_pid, gpu_count, lock_path):
    assert gpu_count in (4, 8), gpu_count
    worker_fd = os.pidfd_open(worker_pid)
    ready_read, ready_write = os.pipe()
    if os.fork() == 0:
        os.close(ready_read)
        os.setsid()
        # A hook returns before the job; retain the lock until its worker exits.
        with open(os.devnull, "r+") as null:
            for fd in (0, 1, 2):
                os.dup2(null.fileno(), fd)
        try:
            with open(lock_path, "a") as lock:
                mode = fcntl.LOCK_SH if gpu_count == 4 else fcntl.LOCK_EX
                while not select.select([worker_fd], [], [], 0)[0]:
                    try:
                        fcntl.flock(lock, mode | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if select.select([worker_fd], [], [], 1)[0]:
                            return
                else:
                    return
                os.write(ready_write, b"1")
                os.close(ready_write)
                select.select([worker_fd], [], [])
        finally:
            os._exit(0)
    os.close(ready_write)
    os.close(worker_fd)
    try:
        assert os.read(ready_read, 1) == b"1", "B200 GPU lock acquisition failed"
    finally:
        os.close(ready_read)


def assert_no_job_containers(gpu_count, runner_name):
    runners = ("b200-oma-8gpu-0", "b200-oma-4gpu-0", "b200-oma-4gpu-1")
    assert runner_name in runners, runner_name
    if gpu_count == 4:
        runners = ("b200-oma-8gpu-0", runner_name)
    command = ["docker", "ps", "--quiet"]
    for runner in runners:
        command.extend(["--filter", f"volume=/data/miles_ci/runner_{runner}"])
    # A killed worker can leave daemon-owned job containers after its lock dies.
    containers = subprocess.check_output(command, text=True).split()
    assert (
        not containers
    ), f"Overlapping B200 job containers remain: {containers}; inspect and clean them up before retrying"


if __name__ == "__main__":
    count = int(os.environ["B200_GPU_COUNT"])
    print(f"Waiting for B200 {'shared half-node' if count == 4 else 'exclusive whole-node'} lock", flush=True)
    acquire(find_worker_pid(), count, "/data/miles_ci/b200-job.lock")
    assert_no_job_containers(count, os.environ["HOSTNAME"])
    print("B200 GPU lock acquired; held through job cleanup", flush=True)
