from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["sglang"], hardware=["hopper"], nightly=True)

import glob
import os
import socket
import subprocess
import sys
from pathlib import Path

_WORLD_SIZE = 4
_NUM_EXPERTS = 64
_HIDDEN = 4096
_MAX_DISPATCH_TOKENS_PER_RANK = 128

_VARIANTS = {
    "sglang_default": {},
    "ib_debug": {"NVSHMEM_DEBUG": "INFO", "NVSHMEM_DEBUG_SUBSYS": "INIT,TRANSPORT"},
    "no_remote_transport": {"NVSHMEM_REMOTE_TRANSPORT": "none"},
    "ib_disabled": {"NVSHMEM_DISABLE_IB": "1", "NVSHMEM_IB_ENABLE_IBGDA": "0"},
}


def _sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120)
    print(result.stdout + result.stderr, flush=True)


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return f"<{error.strerror}>"


def dump_environment() -> None:
    print("=" * 30 + " host " + "=" * 30, flush=True)
    print(f"hostname={socket.gethostname()} runner={os.environ.get('RUNNER_NAME')}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    _sh("nvidia-smi -L; nvidia-smi --query-gpu=index,pci.bus_id,serial,memory.used --format=csv")
    _sh("nvidia-smi topo -m")
    _sh("ulimit -l; df -h /dev/shm | tail -1; uname -r; cat /proc/driver/nvidia/version | head -1")
    _sh("lsmod 2>/dev/null | grep -E 'nvidia_peermem|ib_core|mlx5|gdrdrv' || echo 'lsmod unavailable or no matching modules'")

    print("=" * 30 + " infiniband " + "=" * 30, flush=True)
    _sh("ls -la /dev/infiniband 2>&1")
    for device in sorted(glob.glob("/sys/class/infiniband/*")):
        for port in sorted(glob.glob(f"{device}/ports/*")):
            print(
                f"{os.path.basename(device)} port{os.path.basename(port)}"
                f" state={_read(f'{port}/state')} phys={_read(f'{port}/phys_state')}"
                f" layer={_read(f'{port}/link_layer')} rate={_read(f'{port}/rate')}"
                f" lid={_read(f'{port}/lid')} sm_lid={_read(f'{port}/sm_lid')}"
                f" gid0={_read(f'{port}/gids/0')}",
                flush=True,
            )
    _sh("ibv_devinfo 2>&1 | head -80 || true")
    _sh("rdma link 2>&1 || true")
    _sh("env | grep -E '^(NVSHMEM|NCCL|DEEPEP|SGLANG_DEEPEP)' | sort || true")


def _worker(rank: int, port: int) -> None:
    import deep_ep
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=_WORLD_SIZE, device_id=torch.device("cuda", rank)
    )
    group = dist.new_group(list(range(_WORLD_SIZE)))
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        _MAX_DISPATCH_TOKENS_PER_RANK, _HIDDEN, _WORLD_SIZE, _NUM_EXPERTS
    )
    buffer = deep_ep.Buffer(
        group,
        num_nvl_bytes=int(1e9),
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=_NUM_EXPERTS // _WORLD_SIZE,
    )
    x = torch.randn(_MAX_DISPATCH_TOKENS_PER_RANK, _HIDDEN, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.randint(0, _NUM_EXPERTS, (_MAX_DISPATCH_TOKENS_PER_RANK, 8), device="cuda", dtype=torch.int64)
    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x, topk_idx, _MAX_DISPATCH_TOKENS_PER_RANK, _NUM_EXPERTS, use_fp8=False
    )
    torch.cuda.synchronize()
    dist.barrier()
    print(f"[rank {rank}] deep_ep low-latency buffer + dispatch OK, recv_count={int(recv_count.sum())}", flush=True)
    dist.destroy_process_group()


def run_variant(name: str) -> int:
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=_WORLD_SIZE, join=True)
    print(f"VARIANT {name}: PASS", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--variant":
        return run_variant(sys.argv[2])

    dump_environment()
    results = {}
    for name, extra_env in _VARIANTS.items():
        print("=" * 30 + f" variant {name} {extra_env} " + "=" * 30, flush=True)
        try:
            result = subprocess.run(
                [sys.executable, __file__, "--variant", name], env={**os.environ, **extra_env}, timeout=600
            )
            results[name] = "PASS" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
        except subprocess.TimeoutExpired:
            results[name] = "TIMEOUT"
    print("=" * 30 + " summary " + "=" * 30, flush=True)
    for name, outcome in results.items():
        print(f"PROBE {name}: {outcome}", flush=True)
    return 0 if results["sglang_default"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
