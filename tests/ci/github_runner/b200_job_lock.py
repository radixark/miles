import fcntl
import json
import os
import re
import select
import subprocess
from contextlib import ExitStack
from pathlib import Path


GPU_COUNTS = tuple(range(1, 9))


def find_worker_pid():
    pid = os.getppid()
    while pid > 1:
        proc = Path(f"/proc/{pid}")
        if (proc / "comm").read_text().strip() == "Runner.Worker":
            return pid
        status = (proc / "status").read_text().splitlines()
        pid = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
    raise RuntimeError("B200 job hook must run beneath Runner.Worker")


def running_job_containers(root):
    output = subprocess.check_output(["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Mounts}}"], text=True)
    containers = {}
    for line in output.splitlines():
        container, _, mounts = line.partition(" ")
        for mount in mounts.split(","):
            path = Path(mount)
            if path.parent == root and re.fullmatch(r"runner_b200-oma-[1-8]gpu-\d+", path.name):
                runner = path.name.removeprefix("runner_")
                assert runner not in containers, f"Multiple running job containers for {runner}"
                containers[runner] = container
    return containers


def allocate_gpus(root, gpu_count, runner_name, stack):
    leases = root / "b200-gpu-leases"
    leases.mkdir(exist_ok=True)
    with (leases / "allocation.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        containers = running_job_containers(root)
        assert runner_name not in containers, f"Previous job container remains for {runner_name}: {containers}"
        occupied = {}
        for runner, container in containers.items():
            lease = leases / f"{runner}.json"
            assert lease.exists(), f"Running B200 job has no GPU lease: {runner} {container}"
            occupied[runner] = json.loads(lease.read_text())

        with ExitStack() as attempt:
            available = []
            for gpu in range(8):
                lock = attempt.enter_context((leases / f"gpu-{gpu}.lock").open("a"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                available.append((gpu, lock))

            free = {gpu for gpu, _ in available}
            orphans = {runner: gpus for runner, gpus in occupied.items() if free.intersection(gpus)}
            orphan_gpus = {gpu for gpus in orphans.values() for gpu in gpus}
            assert gpu_count <= 8 - len(orphan_gpus), (
                f"Orphaned B200 job containers block {gpu_count} GPUs: {orphans}; "
                "inspect and clean them up before retrying"
            )
            usable = [(gpu, lock) for gpu, lock in available if gpu not in orphan_gpus]
            if len(usable) < gpu_count:
                return None
            selected = usable[:gpu_count]
            # Duplicate only selected locks into the holder's lifetime.
            for _, lock in selected:
                stack.enter_context(os.fdopen(os.dup(lock.fileno()), "a"))
            gpus = [gpu for gpu, _ in selected]
            (leases / f"{runner_name}.json").write_text(json.dumps(gpus) + "\n")
            return gpus


def acquire(worker_pid, gpu_count, root, runner_name):
    assert gpu_count in GPU_COUNTS, gpu_count
    assert re.fullmatch(rf"b200-oma-{gpu_count}gpu-\d+", runner_name), runner_name
    worker_fd = os.pidfd_open(worker_pid)
    ready_read, ready_write = os.pipe()
    if os.fork() == 0:
        os.close(ready_read)
        os.setsid()
        with open(os.devnull, "r+") as null:
            for fd in (0, 1, 2):
                os.dup2(null.fileno(), fd)
        try:
            with ExitStack() as stack:
                # Keep old whole-node workflow revisions excluded during rollout.
                host_lock = stack.enter_context((root / "b200-job.lock").open("a"))
                mode = fcntl.LOCK_EX if gpu_count == 8 else fcntl.LOCK_SH
                while not select.select([worker_fd], [], [], 0)[0]:
                    try:
                        fcntl.flock(host_lock, mode | fcntl.LOCK_NB)
                    except BlockingIOError:
                        select.select([worker_fd], [], [], 1)
                        continue
                    gpus = allocate_gpus(root, gpu_count, runner_name, stack)
                    if gpus is not None:
                        os.write(ready_write, json.dumps({"gpus": gpus}).encode())
                        os.close(ready_write)
                        ready_write = None
                        select.select([worker_fd], [], [])
                        return
                    select.select([worker_fd], [], [], 1)
        except Exception as error:
            if ready_write is not None:
                os.write(ready_write, json.dumps({"error": str(error)}).encode())
        finally:
            os._exit(0)
    os.close(ready_write)
    os.close(worker_fd)
    with os.fdopen(ready_read) as ready:
        result = json.load(ready)
    assert "error" not in result, result
    return result["gpus"]


if __name__ == "__main__":
    count = int(os.environ["B200_GPU_COUNT"])
    print(f"Waiting for {count} free B200 GPUs", flush=True)
    gpus = acquire(find_worker_pid(), count, Path("/data/miles_ci"), os.environ["HOSTNAME"])
    visible = ",".join(map(str, gpus))
    with open(os.environ["GITHUB_ENV"], "a") as environment:
        environment.write(f"CUDA_VISIBLE_DEVICES={visible}\n")
    print(f"B200 GPUs acquired: {visible}; held through job cleanup", flush=True)
