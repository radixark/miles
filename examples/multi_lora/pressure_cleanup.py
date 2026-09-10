"""Reap only a finished pressure trial's processes on its dedicated Ray cluster."""

import os
import time
from pathlib import Path

import psutil
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _owned_processes(job_id, source_root):
    """Require both Ray job identity and this experiment's source checkout."""
    assert job_id and source_root
    owned = []
    for process in psutil.process_iter():
        try:
            if process.pid == os.getpid() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            environment = process.environ()
            if environment.get("RAY_JOB_ID") == job_id and source_root in environment.get("PYTHONPATH", "").split(
                os.pathsep
            ):
                owned.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return owned


def _reap_on_node(job_id, source_root):
    terminated = set()
    deadline = time.monotonic() + 60
    while processes := _owned_processes(job_id, source_root):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"trial {job_id} still owns processes: {[p.pid for p in processes]}")
        for process in processes:
            try:
                process.terminate()
                terminated.add(process.pid)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=5)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
    return {"node": ray.util.get_node_ip_address(), "terminated_pids": sorted(terminated)}


def reap_trial_processes(job_id, *, ray_address):
    source_root = str(Path(__file__).resolve().parents[2])
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
    reap = ray.remote(num_cpus=0, runtime_env={"env_vars": {"PYTHONPATH": source_root}})(_reap_on_node)
    calls = [
        reap.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(
            job_id, source_root
        )
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("GPU", 0) > 0
    ]
    assert calls, "no live GPU nodes available to verify trial cleanup"
    return ray.get(calls, timeout=120)
