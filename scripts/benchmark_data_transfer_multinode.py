#!/usr/bin/env python3
"""
Multi-node benchmark: put on worker node (72), get on head node (70).

Requires:
  - Ray cluster (head on 70, worker on 72)
  - Mooncake master on 70, MOONCAKE_MASTER=192.168.22.70:50051
  - Both nodes: /root/miles, /root/sglang-venv

Usage:
  cd /root/miles
  export MOONCAKE_MASTER=192.168.22.70:50051
  export MOONCAKE_TE_META_DATA_SERVER="http://192.168.22.70:8080/metadata"
  export MC_STORE_MEMCPY=true
  export MOONCAKE_PROTOCOL=rdma   # or tcp; rdma requires InfiniBand/RoCE
  /root/sglang-venv/bin/python scripts/benchmark_data_transfer_multinode.py --data-size-mb 100
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def make_mock_rollout_data(
    batch_size: int,
    seq_len: int,
    n_samples_per_prompt: int = 4,
    use_routing_replay: bool = False,
    num_layers: int = 64,
    moe_router_topk: int = 2,
) -> dict:
    """Create mock rollout data."""
    num_samples = batch_size * n_samples_per_prompt
    total_lengths = [seq_len + np.random.randint(100, 500) for _ in range(num_samples)]
    response_lengths = [np.random.randint(100, 500) for _ in range(num_samples)]
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
        data["rollout_routed_experts"] = [np.random.randint(0, 8, size=(tot_len - 1, num_layers, moe_router_topk), dtype=np.int32) for tot_len in total_lengths]
    return data


def get_serialized_size(data: dict) -> int:
    return len(pickle.dumps(data))


def _get_node_ids():
    """Return (head_node_id, worker_node_id) where head=70, worker=72."""
    nodes = ray.nodes()
    head_id, worker_id = None, None
    for n in nodes:
        if not n.get("Alive"):
            continue
        addr = n.get("NodeManagerAddress") or (n.get("raylet", {}) or {}).get("node_manager_address", "")
        if "192.168.22.70" in str(addr):
            head_id = n["NodeID"]
        elif "192.168.22.72" in str(addr):
            worker_id = n["NodeID"]
    if not head_id or not worker_id:
        raise RuntimeError("Need both 192.168.22.70 and 192.168.22.72 in cluster")
    return head_id, worker_id


@ray.remote
def put_ray(data: dict):
    """Put on worker (72). Returns ObjectRef for get on head (70)."""
    return ray.put(data)


@ray.remote
class MooncakePutActor:
    """Long-lived Mooncake client on 72 for put. Simulates production rollout worker."""

    def __init__(self):
        from miles.utils.data_transfer import MooncakeDataTransfer

        self.backend = MooncakeDataTransfer(enable_auto_cleanup=False)

    def put(self, data: dict) -> str:
        return self.backend.put(data)


def resolve_data_size_mb(data_size_mb: float) -> tuple[int, int]:
    """Return (batch_size, seq_len) to achieve ~data_size_mb (same as single-node benchmark)."""
    if data_size_mb <= 0:
        return 16, 2048
    scale = (data_size_mb / 1.73) ** 0.5
    batch_size = max(16, int(16 * scale))
    seq_len = max(2048, int(2048 * scale))
    return batch_size, seq_len


def run_multinode_benchmark(
    data_size_mb: float,
    num_rounds: int,
    use_routing_replay: bool = False,
    num_layers: int = 64,
    moe_router_topk: int = 2,
):
    # Workers inherit env from ray start (PYTHONPATH, MOONCAKE_* set on both nodes)
    ray.init(address="auto")
    head_id, worker_id = _get_node_ids()

    batch_size, seq_len = resolve_data_size_mb(data_size_mb)
    data = make_mock_rollout_data(
        batch_size,
        seq_len,
        use_routing_replay=use_routing_replay,
        num_layers=num_layers,
        moe_router_topk=moe_router_topk,
    )
    data_size = get_serialized_size(data) / (1024 * 1024)

    protocol = os.environ.get("MOONCAKE_PROTOCOL", "tcp")
    mooncake_env = {"MOONCAKE_PROTOCOL": protocol}
    print("\nMulti-node benchmark: put@72 -> get@70")
    print("  Ray: driver ray.get(ref); Mooncake: driver backend.get(key) direct (no actor return)")
    print(f"Mooncake protocol: {protocol} (MOONCAKE_PROTOCOL=rdma for RDMA)")
    if use_routing_replay:
        print(f"Routing replay: enabled (num_layers={num_layers}, moe_router_topk={moe_router_topk})")
    print(f"Data: {data_size:.2f} MB (batch={batch_size}, seq_len={seq_len})")
    print("=" * 70)

    put_on_72 = NodeAffinitySchedulingStrategy(node_id=worker_id, soft=False)
    get_on_70 = NodeAffinitySchedulingStrategy(node_id=head_id, soft=False)

    results = {}

    # --- Ray ---
    # put on 72, get in driver (70) - ray.get(ref) triggers 72->70 transfer
    put_fn = put_ray.options(scheduling_strategy=put_on_72)
    put_ms_list, get_ms_list = [], []
    for i in range(num_rounds):
        t0 = time.perf_counter()
        try:
            ref_future = put_fn.remote(data)
            ref = ray.get(ref_future)  # ObjectRef, data lives on 72
            put_ms = (time.perf_counter() - t0) * 1000
            put_ms_list.append(put_ms)
        except Exception as e:
            print(f"  Ray put round {i}: {e}")
            continue
        t0 = time.perf_counter()
        try:
            ray.get(ref)  # driver on 70 fetches from 72
            get_ms = (time.perf_counter() - t0) * 1000
            get_ms_list.append(get_ms)
        except Exception as e:
            print(f"  Ray get round {i}: {e}")
    if put_ms_list and get_ms_list:
        results["ray"] = {
            "put": np.mean(put_ms_list),
            "put_std": np.std(put_ms_list),
            "get": np.mean(get_ms_list),
            "get_std": np.std(get_ms_list),
        }

    # --- Mooncake ---
    # put via actor on 72; get via driver direct (driver on 70, same as Ray)
    # Driver direct get avoids actor-return overhead, comparable to Ray's ray.get(ref)
    runtime_env = {"env_vars": mooncake_env}
    PutActor = MooncakePutActor.options(scheduling_strategy=put_on_72, runtime_env=runtime_env)
    put_actor = PutActor.remote()

    from miles.utils.data_transfer import MooncakeDataTransfer

    mc_backend = MooncakeDataTransfer(enable_auto_cleanup=False)

    put_ms_list, get_ms_list = [], []
    # Warmup
    for _ in range(2):
        try:
            key = ray.get(put_actor.put.remote(data))
            mc_backend.get(key, auto_cleanup=False)
            mc_backend.cleanup(key)
        except Exception:
            pass

    for i in range(num_rounds):
        t0 = time.perf_counter()
        try:
            key_ref = put_actor.put.remote(data)
            key = ray.get(key_ref)
            put_ms = (time.perf_counter() - t0) * 1000
            put_ms_list.append(put_ms)
        except Exception as e:
            print(f"  Mooncake put round {i}: {e}")
            continue
        t0 = time.perf_counter()
        try:
            mc_backend.get(key, auto_cleanup=False)
            mc_backend.cleanup(key)
            get_ms = (time.perf_counter() - t0) * 1000
            get_ms_list.append(get_ms)
        except Exception as e:
            print(f"  Mooncake get round {i}: {e}")
    if put_ms_list and get_ms_list:
        results["mooncake"] = {
            "put": np.mean(put_ms_list),
            "put_std": np.std(put_ms_list),
            "get": np.mean(get_ms_list),
            "get_std": np.std(get_ms_list),
        }

    print("\n" + "=" * 70)
    print("Summary (mean ± std ms)")
    print("-" * 70)
    print(f"{'Backend':<12} {'Put (ms)':<22} {'Get (ms)':<22} Size (MB)")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<12} {r['put']:.1f} ± {r['put_std']:.1f}       {r['get']:.1f} ± {r['get_std']:.1f}       {data_size:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-size-mb", type=float, default=100)
    parser.add_argument("--num-rounds", type=int, default=10)
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
    run_multinode_benchmark(
        args.data_size_mb,
        args.num_rounds,
        use_routing_replay=args.routing_replay,
        num_layers=args.num_layers,
        moe_router_topk=args.moe_router_topk,
    )


if __name__ == "__main__":
    main()
