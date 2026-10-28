"""Serve the two frozen reasoning_gym teachers with SGLang for multi-teacher OPD scoring.

One SGLang server per teacher. A teacher given several GPUs runs data-parallel
replicas (--dp-size N, TP1): scoring is prefill-only, and a 3B-active MoE gains
nothing from tensor parallelism at these lengths. Each task's teacher listens on a
fixed port, --base-port plus its index in TASKS (30000 caesar_cipher, 30001
simple_geometry), which is where scripts/run_qwen3_6_35b_a3b_mopd.py sends it.

Run this on a node that does NOT start the Miles launcher: the launcher's
cleanup pkills every sglang process on its own node. The checkpoints come from
the launcher's `prepare` entrypoint (teacher-<task> under --models-dir).

GPUs per layout of the launcher (node B):
    b200-sync    --teacher caesar_cipher=0,1,2,3 --teacher simple_geometry=4,5,6,7
    b200-async   --teacher caesar_cipher=6 --teacher simple_geometry=7       (GPUs 0-5: rollout)
    gb300-sync   --teacher caesar_cipher=0,1 --teacher simple_geometry=2,3
    gb300-async  --teacher caesar_cipher=2 --teacher simple_geometry=3       (GPUs 0-1: rollout)

Example (b200-sync):
    python examples/on_policy_distillation/mopd_reasoning_gym/serve_teachers.py --models-dir /root/models \
        --teacher caesar_cipher=0,1,2,3 --teacher simple_geometry=4,5,6,7
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TASKS = ("caesar_cipher", "simple_geometry")


def _healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health_generate", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--teacher", action="append", required=True, help="TASK=GPU[,GPU...]")
    parser.add_argument("--base-port", type=int, default=30000)
    parser.add_argument("--log-dir", default=".")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--extra-args", default="", help="Extra sglang.launch_server flags for every teacher")
    args = parser.parse_args()

    processes = []

    def stop(*_):
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        sys.exit(1)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    ports = {}
    for spec in args.teacher:
        name, gpus = spec.split("=")
        port = args.base_port + TASKS.index(name)
        ports[name] = port
        command = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", str(Path(args.models_dir) / f"teacher-{name}"),
            "--host", "0.0.0.0", "--port", str(port),
            "--tp-size", "1", "--dp-size", str(len(gpus.split(","))),
            "--context-length", "32768",
            "--mem-fraction-static", str(args.mem_fraction_static),
            # n=1 per prompt: no prefix worth caching, and the hybrid model needs extra state to cache.
            "--disable-radix-cache",
            "--enable-metrics",
            *args.extra_args.split(),
        ]  # fmt: skip
        log = open(Path(args.log_dir) / f"teacher-{name}.log", "wb")
        print(f"[{name}] port {port} gpus {gpus}: {' '.join(command)}", flush=True)
        processes.append(
            subprocess.Popen(
                command,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )
        # Servers started together race for the same internal dist-init ports.
        time.sleep(20)

    deadline = time.monotonic() + 1200
    while not all(_healthy(port) for port in ports.values()):
        if any(process.poll() is not None for process in processes):
            print("A teacher exited during startup; see teacher-*.log", flush=True)
            stop()
        if time.monotonic() > deadline:
            print("Teachers were not healthy within 20 minutes", flush=True)
            stop()
        time.sleep(5)
    print("READY " + " ".join(f"{name}={port}" for name, port in ports.items()), flush=True)

    while all(process.poll() is None for process in processes):
        time.sleep(5)
    print("A teacher exited", flush=True)
    stop()


if __name__ == "__main__":
    main()
