#!/usr/bin/env python3
"""
Launch sglang-diffusion workers, start the Miles DiffusionRouter, then run
the sglang diffusion serving benchmark against the router.

Example:
  python examples/diffusion_router/bench_router.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 4
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import requests


def _require_non_empty_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError(
            "--model must be a non-empty model ID/path. "
            "Detected an empty value, which often means a shell variable such as "
            "$MODEL was unset."
        )
    return normalized


def _wait_for_health(
    url: str, timeout: int, label: str, proc: subprocess.Popen | None = None,
) -> None:
    start = time.time()
    last_print = 0.0
    while True:
        elapsed = time.time() - start

        # Fail fast if the backing process has already exited
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{label} process exited with code {proc.returncode}. "
                "Run the worker command directly to see the error."
            )

        try:
            resp = requests.get(f"{url}/health", timeout=1)
            if resp.status_code == 200:
                print(f"  [bench] {label} is healthy ({elapsed:.0f}s)", flush=True)
                return
        except requests.RequestException:
            pass

        if elapsed - last_print >= 30:
            print(f"  [bench] Still waiting for {label}... ({elapsed:.0f}s elapsed)", flush=True)
            last_print = elapsed

        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for {label} at {url}.")
        time.sleep(1)


def _build_worker_urls(host: str, base_port: int, count: int, stride: int) -> list[str]:
    return [f"http://{host}:{base_port + i * stride}" for i in range(count)]


def _infer_client_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def _is_port_available(host: str, port: int) -> bool:
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _reserve_available_port(host: str, preferred_port: int, used_ports: set[int]) -> int:
    if preferred_port < 1 or preferred_port > 65535:
        raise ValueError(f"Invalid port: {preferred_port}")

    for port in range(preferred_port, 65536):
        if port in used_ports:
            continue
        if _is_port_available(host, port):
            used_ports.add(port)
            return port

    for port in range(1024, preferred_port):
        if port in used_ports:
            continue
        if _is_port_available(host, port):
            used_ports.add(port)
            return port

    raise RuntimeError(
        f"Unable to reserve a free port for host {host}. "
        f"Preferred start={preferred_port}."
    )


def _parse_gpu_id_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _detect_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _resolve_gpu_pool(args: argparse.Namespace, env: dict[str, str]) -> list[str] | None:
    if args.worker_gpu_ids:
        return [str(x) for x in args.worker_gpu_ids]

    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        parsed = _parse_gpu_id_list(visible)
        if parsed:
            return parsed

    gpu_count = _detect_gpu_count()
    if gpu_count > 0:
        return [str(i) for i in range(gpu_count)]
    return None


def _terminate_all(processes: Iterable[subprocess.Popen]) -> None:
    procs = list(processes)

    def _signal_group(proc: subprocess.Popen, sig: int) -> None:
        try:
            # Processes are launched with start_new_session=True so each has its own group.
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            if proc.poll() is None:
                try:
                    os.kill(proc.pid, sig)
                except ProcessLookupError:
                    pass

    for proc in procs:
        _signal_group(proc, signal.SIGTERM)

    for proc in procs:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)

    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Miles DiffusionRouter with sglang bench_serving.")
    parser.add_argument("--model", type=str, required=True, help="Diffusion model HF ID or local path.")
    parser.add_argument("--router-host", type=str, default="127.0.0.1", help="Router bind host.")
    parser.add_argument("--router-port", type=int, default=30080, help="Router port.")
    parser.add_argument("--routing-algorithm", type=str, default="least-request",
                        choices=["least-request", "round-robin", "random"],
                        help="Load-balancing algorithm for the router.")
    parser.add_argument("--router-verbose", action="store_true", help="Enable router verbose logging.")
    parser.add_argument("--router-extra-args", type=str, default="", help="Extra args for the router demo script.")

    parser.add_argument("--worker-host", type=str, default="127.0.0.1", help="Worker bind host.")
    parser.add_argument("--worker-urls", nargs="*", default=[], help="Existing worker URLs to use.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers to launch.")
    parser.add_argument("--worker-base-port", type=int, default=10090, help="Base port for launched workers.")
    parser.add_argument(
        "--worker-port-stride",
        type=int,
        default=2,
        help="Port increment between launched workers. Keep >=2 to avoid sglang internal port collisions.",
    )
    parser.add_argument(
        "--worker-master-port-base",
        type=int,
        default=30005,
        help="Base torch distributed master port for launched workers.",
    )
    parser.add_argument(
        "--worker-scheduler-port-base",
        type=int,
        default=5555,
        help="Base scheduler port for launched workers.",
    )
    parser.add_argument(
        "--worker-internal-port-stride",
        type=int,
        default=1000,
        help=(
            "Stride used between workers for master/scheduler base ports. "
            "Use >= 101 because sglang randomizes each by +[0,100]."
        ),
    )
    parser.add_argument("--num-gpus-per-worker", type=int, default=1, help="GPUs per worker.")
    parser.add_argument(
        "--worker-gpu-ids",
        nargs="*",
        default=None,
        help=(
            "Optional GPU IDs/UUIDs for launched workers. They are consumed in order, "
            "in groups of --num-gpus-per-worker."
        ),
    )
    parser.add_argument("--worker-extra-args", type=str, default="", help="Extra args for `sglang serve`.")
    parser.add_argument("--skip-workers", action="store_true", help="Do not launch workers.")

    parser.add_argument("--dataset", type=str, default="random", choices=["vbench", "random"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--bench-extra-args", type=str, default="", help="Extra args for bench_serving.")

    parser.add_argument("--wait-timeout", type=int, default=1200, help="Seconds to wait for services to be healthy.")

    args = parser.parse_args()
    args.model = _require_non_empty_model(args.model)

    try:
        import sglang  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "sglang is not installed.\n"
            "Install with:  uv pip install \"sglang[diffusion]\" --prerelease=allow"
        )
    env = dict(os.environ)

    worker_urls = list(args.worker_urls)
    if not worker_urls:
        if args.worker_port_stride < 1:
            raise ValueError("--worker-port-stride must be >= 1")
        if args.worker_internal_port_stride < 101:
            raise ValueError("--worker-internal-port-stride must be >= 101")
        worker_urls = _build_worker_urls(
            args.worker_host,
            args.worker_base_port,
            args.num_workers,
            args.worker_port_stride,
        )

    if args.skip_workers and not worker_urls:
        raise ValueError("No workers specified. Provide --worker-urls or disable --skip-workers.")

    if not _is_port_available(args.router_host, args.router_port):
        raise RuntimeError(
            f"Router port {args.router_port} on {args.router_host} is already in use. "
            "Stop the existing router/process or change --router-port."
        )

    processes: list[subprocess.Popen] = []
    try:
        if not args.skip_workers:
            reserved_ports: set[int] = {args.router_port}
            worker_internal_ports: list[tuple[int, int]] = []
            for url in worker_urls:
                port = int(url.rsplit(":", 1)[1])
                if not _is_port_available(args.worker_host, port):
                    raise RuntimeError(
                        f"Worker port {port} on {args.worker_host} is already in use. "
                        "Stop existing servers or change --worker-base-port/--worker-port-stride."
                    )
                reserved_ports.add(port)

            for i, _ in enumerate(worker_urls):
                preferred_master = args.worker_master_port_base + i * args.worker_internal_port_stride
                preferred_scheduler = (
                    args.worker_scheduler_port_base + i * args.worker_internal_port_stride
                )
                master_port = _reserve_available_port(args.worker_host, preferred_master, reserved_ports)
                scheduler_port = _reserve_available_port(args.worker_host, preferred_scheduler, reserved_ports)
                worker_internal_ports.append((master_port, scheduler_port))
                if master_port != preferred_master or scheduler_port != preferred_scheduler:
                    print(
                        "[bench] Adjusted internal worker ports due to conflict: "
                        f"worker={i} master={master_port} scheduler={scheduler_port}",
                        flush=True,
                    )

            sglang_bin = Path(sys.executable).resolve().parent / "sglang"
            if sglang_bin.exists():
                sglang_cli_cmd = [str(sglang_bin)]
            else:
                sglang_cli_cmd = [sys.executable, "-c", "from sglang.cli.main import main; main()"]
            gpu_pool = _resolve_gpu_pool(args, env)
            total_gpus_needed = len(worker_urls) * args.num_gpus_per_worker
            if gpu_pool is not None and len(gpu_pool) < total_gpus_needed:
                raise ValueError(
                    f"Need {total_gpus_needed} GPU slots for {len(worker_urls)} worker(s) x "
                    f"{args.num_gpus_per_worker} GPU(s), but only {len(gpu_pool)} visible. "
                    "Set --worker-gpu-ids, reduce --num-workers, or reduce --num-gpus-per-worker."
                )
            for i, url in enumerate(worker_urls):
                port = int(url.rsplit(":", 1)[1])
                master_port, scheduler_port = worker_internal_ports[i]
                cmd = [
                    *sglang_cli_cmd,
                    "serve",
                    "--model-path",
                    args.model,
                    "--num-gpus",
                    str(args.num_gpus_per_worker),
                    "--host",
                    args.worker_host,
                    "--port",
                    str(port),
                    "--master-port",
                    str(master_port),
                    "--scheduler-port",
                    str(scheduler_port),
                ]
                if args.worker_extra_args:
                    cmd += shlex.split(args.worker_extra_args)
                worker_env = env
                if gpu_pool is not None:
                    start = i * args.num_gpus_per_worker
                    end = start + args.num_gpus_per_worker
                    assigned = gpu_pool[start:end]
                    worker_env = dict(env)
                    worker_env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
                    print(
                        f"[bench] Worker {i} uses CUDA_VISIBLE_DEVICES={worker_env['CUDA_VISIBLE_DEVICES']}",
                        flush=True,
                    )
                print(
                    f"[bench] Launching worker {i} on port {port} "
                    f"(master={master_port}, scheduler={scheduler_port})...",
                    flush=True,
                )
                processes.append(subprocess.Popen(cmd, env=worker_env, start_new_session=True))

            if args.max_concurrency and worker_urls:
                per_worker = (args.max_concurrency + len(worker_urls) - 1) // len(worker_urls)
                if per_worker > 1:
                    print(
                        "[bench] Warning: "
                        f"max_concurrency={args.max_concurrency} over {len(worker_urls)} workers "
                        f"can drive up to ~{per_worker} concurrent requests per worker, which may OOM "
                        "large diffusion models.",
                        flush=True,
                    )

            print(f"[bench] Waiting for {len(worker_urls)} worker(s) to become healthy (this may take several minutes)...", flush=True)
            for i, url in enumerate(worker_urls):
                _wait_for_health(url, args.wait_timeout, f"worker {url}", proc=processes[i])

        router_cmd = [
            sys.executable,
            "examples/diffusion_router/demo.py",
            "--host",
            args.router_host,
            "--port",
            str(args.router_port),
            "--worker-urls",
            *worker_urls,
        ]
        if args.routing_algorithm:
            router_cmd += ["--routing-algorithm", args.routing_algorithm]
        if args.router_verbose:
            router_cmd.append("--verbose")
        if args.router_extra_args:
            router_cmd += shlex.split(args.router_extra_args)

        if not _is_port_available(args.router_host, args.router_port):
            raise RuntimeError(
                f"Router port {args.router_port} on {args.router_host} is already in use. "
                "Stop the existing router/process or change --router-port."
            )

        print(f"[bench] Launching router on port {args.router_port}...", flush=True)
        router_proc = subprocess.Popen(router_cmd, start_new_session=True)
        processes.append(router_proc)

        router_host = _infer_client_host(args.router_host)
        base_url = f"http://{router_host}:{args.router_port}"
        _wait_for_health(base_url, args.wait_timeout, "router", proc=router_proc)

        print(f"[bench] Running benchmark: {args.num_prompts} prompts, concurrency={args.max_concurrency}", flush=True)

        bench_cmd = [
            sys.executable,
            "-m",
            "sglang.multimodal_gen.benchmarks.bench_serving",
            "--base-url",
            base_url,
            "--model",
            args.model,
            "--dataset",
            args.dataset,
            "--num-prompts",
            str(args.num_prompts),
            "--request-rate",
            str(args.request_rate),
            "--log-level",
            args.log_level,
        ]
        if args.dataset_path:
            bench_cmd += ["--dataset-path", args.dataset_path]
        if args.max_concurrency:
            bench_cmd += ["--max-concurrency", str(args.max_concurrency)]
        if args.task:
            bench_cmd += ["--task", args.task]
        if args.width:
            bench_cmd += ["--width", str(args.width)]
        if args.height:
            bench_cmd += ["--height", str(args.height)]
        if args.num_frames:
            bench_cmd += ["--num-frames", str(args.num_frames)]
        if args.fps:
            bench_cmd += ["--fps", str(args.fps)]
        if args.output_file:
            bench_cmd += ["--output-file", args.output_file]
        if args.disable_tqdm:
            bench_cmd.append("--disable-tqdm")
        if args.bench_extra_args:
            bench_cmd += shlex.split(args.bench_extra_args)

        bench_proc = subprocess.Popen(bench_cmd, env=env, start_new_session=True)
        processes.append(bench_proc)
        return bench_proc.wait()
    finally:
        _terminate_all(reversed(processes))


if __name__ == "__main__":
    raise SystemExit(main())
