"""Sample the memory of one run until killed, then print the peaks.

    python tests/manual/output_store_replay/sample_memory.py <out.jsonl> [interval seconds, default 5]

Each line: per trainer rank the /proc/<pid>/status counters in kB (VmHWM is the kernel's high-water mark), the
cgroup's anon + shmem bytes (what page-cache reclaim cannot free), and nvidia-smi's used MiB per GPU.
"""

import glob
import json
import signal
import subprocess
import sys
import time

STATUS_KEYS = ("VmHWM", "VmRSS", "VmPin", "RssAnon", "RssShmem")


def trainer_ranks() -> dict[int, dict[str, int]]:
    ranks = {}
    for cmdline in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            if b"MegatronTrainRayActor" not in open(cmdline, "rb").read():
                continue
            pid = int(cmdline.split("/")[2])
            status = dict(line.split(":", 1) for line in open(f"/proc/{pid}/status"))
            ranks[pid] = {key: int(status[key].split()[0]) for key in STATUS_KEYS if key in status}
        except (OSError, ValueError):
            continue
    return ranks


def cgroup_anon_shmem() -> int | None:
    try:
        stat = dict(line.split() for line in open("/sys/fs/cgroup/memory.stat"))
    except OSError:
        return None
    return int(stat["anon"]) + int(stat["shmem"])


def gpu_used_mib() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
    )
    return [int(value) for value in out.stdout.split()]


def main(path: str, interval: float) -> None:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    hwm, host, gpus = {}, 0, []
    try:
        with open(path, "a") as out:
            while True:
                sample = {
                    "t": time.time(),
                    "ranks": trainer_ranks(),
                    "host": cgroup_anon_shmem(),
                    "gpu": gpu_used_mib(),
                }
                out.write(json.dumps(sample) + "\n")
                out.flush()
                for pid, status in sample["ranks"].items():
                    hwm[pid] = max(hwm.get(pid, 0), status.get("VmHWM", 0))
                host = max(host, sample["host"] or 0)
                gpus = [max(pair) for pair in zip(gpus or sample["gpu"], sample["gpu"], strict=True)]
                time.sleep(interval)
    finally:
        print(f"trainer rank VmHWM GB: {[round(kb * 1024 / 1e9, 1) for kb in hwm.values()]}")
        print(f"cgroup anon + shmem peak GB: {round(host / 1e9, 1)}")
        print(f"GPU used peak MiB: {gpus}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0)
