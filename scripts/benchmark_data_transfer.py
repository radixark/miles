#!/usr/bin/env python3
"""
Benchmark script for rollout data transfer backends: Ray vs Mooncake vs Disk.

Compares put/get latency and throughput for mock rollout data similar to real
training workloads. Run with:

  # All three backends (Ray + Mooncake + Disk):
  ray start --head
  export MOONCAKE_MASTER=127.0.0.1:50051
  export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
  export MC_STORE_MEMCPY=true
  python scripts/benchmark_data_transfer.py --backends ray mooncake disk --data-size-mb 100
  # Note: use a venv with Ray installed (e.g. /root/sglang-venv/bin/python)

  # Ray + Disk only (no Mooncake server):
  python scripts/benchmark_data_transfer.py --backends ray disk
"""

import argparse
import pickle
import sys
import tempfile
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def make_mock_rollout_data(
    batch_size: int = 16,
    seq_len: int = 2048,
    n_samples_per_prompt: int = 4,
    use_routing_replay: bool = False,
    num_layers: int = 64,
    moe_router_topk: int = 2,
) -> dict:
    """Create mock rollout data resembling real training data structure."""
    num_samples = batch_size * n_samples_per_prompt
    total_lengths = [seq_len + np.random.randint(100, 500) for _ in range(num_samples)]
    response_lengths = [np.random.randint(100, 500) for _ in range(num_samples)]

    # tokens: list of arrays, one per sample
    tokens = [np.random.randint(0, 32000, size=l, dtype=np.int32) for l in total_lengths]
    loss_masks = [[1] * (resp_len - 50) + [0] * 50 for resp_len in response_lengths]
    rollout_log_probs = [np.random.randn(tot_len - resp_len).astype(np.float32).tolist() for tot_len, resp_len in zip(total_lengths, response_lengths)]

    data = {
        "partition": list(range(num_samples)),
        "tokens": tokens,
        "response_lengths": response_lengths,
        "rewards": [1.0] * num_samples,
        "loss_masks": loss_masks,
        "rollout_log_probs": rollout_log_probs,
        "total_lengths": total_lengths,
    }
    if use_routing_replay:
        # rollout_routed_experts: list of (seq_len-1, num_layers, topk) int32 per sample
        data["rollout_routed_experts"] = [np.random.randint(0, 8, size=(tot_len - 1, num_layers, moe_router_topk), dtype=np.int32) for tot_len in total_lengths]
    return data


def get_serialized_size(data: dict) -> int:
    """Get size in bytes of serialized data."""
    return len(pickle.dumps(data))


def benchmark_backend(backend_name: str, backend, data: dict, num_rounds: int = 20) -> dict:
    """Run put/get benchmark for a backend."""
    results = {"put_ms": [], "get_ms": [], "errors": []}

    # Mooncake: warmup to complete init (segment mount, buffer reg, etc.)
    if backend_name == "mooncake":
        for _ in range(2):
            try:
                h = backend.put(data)
                backend.get(h, auto_cleanup=False)
                backend.cleanup(h)
            except Exception:
                pass

    for i in range(num_rounds):
        # Put
        t0 = time.perf_counter()
        try:
            handle = backend.put(data)
            put_ms = (time.perf_counter() - t0) * 1000
            results["put_ms"].append(put_ms)
        except Exception as e:
            results["errors"].append(f"put round {i}: {e}")
            continue

        # Get
        t0 = time.perf_counter()
        try:
            retrieved = backend.get(handle)
            get_ms = (time.perf_counter() - t0) * 1000
            results["get_ms"].append(get_ms)
        except Exception as e:
            results["errors"].append(f"get round {i}: {e}")
            continue

        # Cleanup if backend supports it
        if hasattr(backend, "cleanup") and backend_name != "ray":
            try:
                backend.cleanup(handle)
            except Exception:
                pass

    return results


def run_benchmarks(
    backends: list[str],
    batch_size: int,
    seq_len: int,
    num_rounds: int,
    use_routing_replay: bool = False,
    num_layers: int = 64,
    moe_router_topk: int = 2,
):
    """Run benchmarks for specified backends."""
    data = make_mock_rollout_data(
        batch_size=batch_size,
        seq_len=seq_len,
        use_routing_replay=use_routing_replay,
        num_layers=num_layers,
        moe_router_topk=moe_router_topk,
    )
    data_size_mb = get_serialized_size(data) / (1024 * 1024)

    print(f"\nMock rollout data: batch_size={batch_size}, seq_len={seq_len}")
    if use_routing_replay:
        print(f"Routing replay: enabled (num_layers={num_layers}, moe_router_topk={moe_router_topk})")
    print(f"Serialized size: {data_size_mb:.2f} MB")
    print("=" * 70)

    all_results = {}

    for name in backends:
        print(f"\nBackend: {name}")
        try:
            if name == "ray":
                import ray
                from miles.utils.data_transfer import RayDataTransfer

                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                backend = RayDataTransfer()
            elif name == "disk":
                # Import DiskDataTransfer without pulling in Ray
                from miles.utils.data_transfer import DiskDataTransfer

                tmpdir = tempfile.mkdtemp(prefix="miles_bench_disk_")
                backend = DiskDataTransfer(base_dir=tmpdir)
            elif name == "mooncake":
                from miles.utils.data_transfer import MooncakeDataTransfer

                backend = MooncakeDataTransfer(enable_auto_cleanup=False)
            else:
                print(f"  Unknown backend: {name}")
                continue

            results = benchmark_backend(name, backend, data, num_rounds=num_rounds)
            all_results[name] = results

            if results["errors"]:
                print(f"  Errors: {results['errors'][:3]}...")
                continue

            put_arr = np.array(results["put_ms"])
            get_arr = np.array(results["get_ms"])

            put_mean = np.mean(put_arr)
            put_std = np.std(put_arr)
            get_mean = np.mean(get_arr)
            get_std = np.std(get_arr)
            throughput_put = data_size_mb / (put_mean / 1000) if put_mean > 0 else 0
            throughput_get = data_size_mb / (get_mean / 1000) if get_mean > 0 else 0

            print(f"  Put:  {put_mean:.2f} ± {put_std:.2f} ms  ({throughput_put:.1f} MB/s)")
            print(f"  Get:  {get_mean:.2f} ± {get_std:.2f} ms  ({throughput_get:.1f} MB/s)")

            # Cleanup disk temp
            if name == "disk" and hasattr(backend, "_base_dir"):
                import shutil

                try:
                    shutil.rmtree(backend._base_dir, ignore_errors=True)
                except Exception:
                    pass

        except ImportError as e:
            print(f"  Skip: {e} (install with: pip install ray)")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    return all_results, data_size_mb


def main():
    parser = argparse.ArgumentParser(description="Benchmark data transfer backends")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["ray", "mooncake", "disk"],
        choices=["ray", "mooncake", "disk"],
        help="Backends to benchmark (default: ray mooncake disk)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Simulated rollout batch size (default: 16)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Simulated sequence length (default: 2048)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=20,
        help="Number of put/get rounds per backend (default: 20)",
    )
    parser.add_argument(
        "--data-size-mb",
        type=float,
        default=None,
        help="Target data size in MB (overrides batch-size/seq-len to achieve ~target). E.g. 100, 1000",
    )
    parser.add_argument(
        "--routing-replay",
        action="store_true",
        help="Include rollout_routed_experts in mock data (MoE R3 / rollout routing replay)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=64,
        help="MoE num_layers for routing replay mock (default: 64)",
    )
    parser.add_argument(
        "--moe-router-topk",
        type=int,
        default=2,
        help="MoE router top-k for routing replay mock (default: 2)",
    )
    args = parser.parse_args()

    # Scale batch_size, seq_len to achieve target data size if specified
    if args.data_size_mb is not None:
        # Calibrated from ~1.73MB at batch=16, seq=2048
        scale = (args.data_size_mb / 1.73) ** 0.5
        args.batch_size = max(16, int(16 * scale))
        args.seq_len = max(2048, int(2048 * scale))
        print(f"Target {args.data_size_mb} MB -> batch_size={args.batch_size}, seq_len={args.seq_len}")

    print("Miles Data Transfer Benchmark: Ray vs Mooncake vs Disk")
    if args.routing_replay:
        print("Mode: with routing replay (rollout_routed_experts)")
    results, data_size_mb = run_benchmarks(
        args.backends,
        args.batch_size,
        args.seq_len,
        args.num_rounds,
        use_routing_replay=args.routing_replay,
        num_layers=args.num_layers,
        moe_router_topk=args.moe_router_topk,
    )

    # Summary table
    if results:
        print("\n" + "=" * 70)
        print("Summary (mean ± std ms)")
        print("-" * 70)
        header = f"{'Backend':<12} {'Put (ms)':<18} {'Get (ms)':<18} {'Size (MB)':<12}"
        print(header)
        print("-" * 70)
        for name, r in results.items():
            if r["put_ms"] and r["get_ms"]:
                put_s = f"{np.mean(r['put_ms']):.1f} ± {np.std(r['put_ms']):.1f}"
                get_s = f"{np.mean(r['get_ms']):.1f} ± {np.std(r['get_ms']):.1f}"
                print(f"{name:<12} {put_s:<18} {get_s:<18} {data_size_mb:.2f}")


if __name__ == "__main__":
    main()
