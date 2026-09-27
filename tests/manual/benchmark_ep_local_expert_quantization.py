"""Standalone routed-expert iterator benchmark; this is not an end-to-end run.

Choose the Miles checkout exclusively through PYTHONPATH, preserving the runtime's
Megatron/TransformerEngine paths. Run the same command against both checkouts::

    PYTHONPATH=/path/to/miles:$PYTHONPATH torchrun --standalone --nproc-per-node=4 \
        /path/to/validation/profile_expert_iterator.py --ep-size 4 --edp-size 1 \
        --label baseline-ep4 --output-dir /path/to/artifacts/iterator-baseline-ep4 \
        --profile --torch-trace

Repeat with --ep-size 2 --edp-size 2. Defaults read the imported checkout's actual
GLM-5.2 model script: two MoE layers, 256 experts, hidden 6144, intermediate 2048.
Only rank 0 materializes output, as in the one-node end-to-end weight sender.
Weights are random BF16 CUDA tensors, deterministically seeded by global name;
expert-DP replicas hold identical tensors. No checkpoint, model forward/backward,
serving engine, transport, global CPU weights, or retained output-bucket list.

Headline timings have no profiling hooks. Profiling runs one additional update
afterward; its wall time is diagnostic only. CPU stage timings overlap/nest and
are not GPU kernel timings. A Torch trace captures CUDA copies and quantization.
"""

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import runpy
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

NVFP4_ENV = {
    "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG": "1",
    "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1",
    "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
}
os.environ.update(NVFP4_ENV)

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

import miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct as direct
from miles.backends.training_utils.parallel import set_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.distributed_utils import get_gloo_group, init_gloo_group


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ep-size", type=int, choices=(2, 4), required=True)
    parser.add_argument("--edp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--buffer-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--profile", action="store_true", help="Separate CPU cProfile/stage-measurement update")
    parser.add_argument("--torch-trace", action="store_true", help="Also capture a rank-0 CPU/CUDA Chrome trace")
    cli = parser.parse_args()
    parser.error("Require EP*EDP=4") if cli.ep_size * cli.edp_size != 4 else None
    if cli.warmup < 1 or cli.iterations < 1 or cli.buffer_bytes < 1:
        parser.error("warmup, iterations, and buffer-bytes must be positive")
    return cli


def _parallel_state(ep_size, edp_size):
    world_size = dist.get_world_size()
    assert world_size == ep_size * edp_size == 4
    rank = dist.get_rank()
    selected = {}
    rank_lists = {
        "ep": [[replica * ep_size + ep for ep in range(ep_size)] for replica in range(edp_size)],
        "edp": [[replica * ep_size + ep for replica in range(edp_size)] for ep in range(ep_size)],
        "tp": [list(range(world_size))],
        "etp": [[r] for r in range(world_size)],
        "pp": [[r] for r in range(world_size)],
    }
    for kind, groups in rank_lists.items():
        for ranks in groups:
            group = dist.new_group(ranks=ranks, backend="nccl")
            if rank in ranks:
                selected[kind] = SimpleNamespace(rank=ranks.index(rank), size=len(ranks), group=group)
    selected["tp_dp_cp"] = selected["tp"]
    state = SimpleNamespace(**selected)
    set_parallel_state(state)
    return state


def _model_args(repo_root, buffer_bytes):
    script = repo_root / "scripts/models/glm5.2-744B-A40B.py"
    definition = runpy.run_path(str(script))
    hidden = definition["NHIDDEN"]
    intermediate = definition["MOE_FFN_HIDDEN"]
    num_experts = definition["MOE_ROUTED_EXPERTS"]
    first_layer = definition["N_DENSE_LAYERS"]
    assert (hidden, intermediate, num_experts, first_layer) == (6144, 2048, 256, 3)
    args = argparse.Namespace(
        sglang_speculative_algorithm=None,
        custom_model_provider_path=None,
        mtp_num_layers=None,
        num_layers=first_layer + 2,
        num_experts=num_experts,
        q_lora_rank=None,  # Routed-only stream has no attention atomic groups.
        vocab_size=154880,
        hidden_size=hidden,
        num_attention_heads=definition["NHEADS"],
        num_query_groups=definition["NHEADS"],
        kv_channels=192,
        swiglu=True,
        update_weight_buffer_size=buffer_bytes,
    )
    return args, intermediate, (first_layer, first_layer + 1), script


def _local_weights(args, intermediate, layers, state, seed):
    local_count = args.num_experts // state.ep.size
    first_expert = state.ep.rank * local_count
    generator = torch.Generator(device="cuda")
    weights = {}
    for layer in layers:
        for expert in range(first_expert, first_expert + local_count):
            for projection, shape in (
                ("linear_fc1", (2 * intermediate, args.hidden_size)),
                ("linear_fc2", (args.hidden_size, intermediate)),
            ):
                name = f"module.module.decoder.layers.{layer}.mlp.experts.{projection}.weight{expert}"
                name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
                generator.manual_seed((seed + name_seed) % (2**63 - 1))
                tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator).mul_(0.02)
                tensor.tensor_model_parallel = False
                tensor.partition_dim = 0 if projection == "linear_fc1" else 1
                tensor.partition_stride = 2 if projection == "linear_fc1" else 1
                weights[name] = tensor
    return weights


def _consume(iterator, weights, *, count_bytes=False):
    tensor_count = 0
    total_bytes = 0
    bucket_count = 0
    for bucket in iterator.iter_hf_weights(weights, materialize=dist.get_rank() == 0):
        tensor_count += len(bucket)
        bucket_count += 1
        if count_bytes:
            total_bytes += sum(tensor.nbytes for _, tensor in bucket)
        del bucket
    return {"tensors": tensor_count, "bytes": total_bytes if count_bytes else None, "buckets": bucket_count}


def _iteration(iterator, weights, *, count_bytes=False):
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    counts = _consume(iterator, weights, count_bytes=count_bytes)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    measurements = [None] * dist.get_world_size()
    dist.all_gather_object(measurements, {"seconds": elapsed, **counts}, group=get_gloo_group())
    assert measurements[0]["tensors"] == 2 * 256 * 9, measurements
    assert all(row["tensors"] == 0 for row in measurements[1:]), measurements
    return {"max_rank_seconds": max(row["seconds"] for row in measurements), "ranks": measurements}


@contextmanager
def _stage_ranges():
    stats = defaultdict(lambda: {"calls": 0, "cpu_seconds": 0.0})

    def wrap(original, label):
        def measured(*args, **kwargs):
            with record_function(label):
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    stats[label]["calls"] += 1
                    stats[label]["cpu_seconds"] += time.perf_counter() - started

        return measured

    targets = [(direct, "convert_to_hf", "expert/convert_to_hf")]
    helper = None
    if hasattr(direct, "ExpertGather"):
        targets.append((direct.ExpertGather, "__call__", "expert/gather_converted"))
        helper = sys.modules[direct.ExpertGather.__module__]
    elif hasattr(direct, "gather_expert_units"):
        targets.append((direct, "gather_expert_units", "expert/gather_converted"))
        helper = sys.modules[direct.gather_expert_units.__module__]
    if helper is not None:
        for name in ("_describe_units", "_pack_units", "_unpack_units"):
            if hasattr(helper, name):
                targets.append((helper, name, "expert/" + name.removeprefix("_")))
    if hasattr(direct, "_materialize_expert_batch"):
        targets.append((direct, "_materialize_expert_batch", "expert/baseline_gather_bf16"))
    for name in ("all_gather_object", "all_gather", "all_gather_into_tensor", "broadcast"):
        targets.append((dist, name, "collective/" + name))
    with ExitStack() as stack:
        for module, name, label in targets:
            stack.enter_context(patch.object(module, name, wrap(getattr(module, name), label)))
        yield stats


def _profile_iteration(cli, iterator, weights):
    rank = dist.get_rank()
    cpu_profile = cProfile.Profile() if rank == 0 else None
    with _stage_ranges() as stage_stats, ExitStack() as stack:
        gpu_profile = None
        if cli.torch_trace and rank == 0:
            gpu_profile = stack.enter_context(profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]))
        if cpu_profile is not None:
            cpu_profile.enable()
        measurement = _iteration(iterator, weights)
        if cpu_profile is not None:
            cpu_profile.disable()
    if rank == 0:
        cpu_profile.dump_stats(str(cli.output_dir / "rank0.prof"))
        text = io.StringIO()
        for sort_by in ("cumulative", "tottime"):
            pstats.Stats(cpu_profile, stream=text).strip_dirs().sort_stats(sort_by).print_stats(60)
        (cli.output_dir / "rank0-cprofile.txt").write_text(text.getvalue())
        if gpu_profile is not None:
            gpu_profile.export_chrome_trace(str(cli.output_dir / "rank0-trace.json"))
            table = gpu_profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
            (cli.output_dir / "rank0-torch-profile.txt").write_text(table)
    all_stats = [None] * dist.get_world_size()
    dist.all_gather_object(all_stats, dict(stage_stats), group=get_gloo_group())
    return {"diagnostic_only": True, "measurement": measurement, "cpu_stage_stats_by_rank": all_stats}


def _provenance(repo_root, model_script):
    paths = [Path(direct.__file__), model_script]
    helper = paths[0].with_name("expert_quantization.py")
    if helper.exists():
        paths.append(helper)
    return {
        "repo": str(repo_root),
        "commit": subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "-C", str(repo_root), "status", "--short"], text=True),
        "file_sha256": {
            str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }


def main():
    cli = _cli()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    init_gloo_group()
    try:
        state = _parallel_state(cli.ep_size, cli.edp_size)
        repo_root = Path(direct.__file__).resolve().parents[4]
        args, intermediate, layers, model_script = _model_args(repo_root, cli.buffer_bytes)
        weights = _local_weights(args, intermediate, layers, state, cli.seed)
        model = torch.nn.Module()
        model.config = SimpleNamespace(mtp_num_layers=None)
        with patch.object(direct, "named_params_and_buffers", lambda _args, _model: iter(weights.items())):
            iterator = direct.HfWeightIteratorDirect(
                args,
                [model],
                placement=WeightUpdatePlacement(gather_pp=False),
                model_name="glmmoedsa",
                quantization_config={"quant_method": "nvfp4"},
            )
        cli.output_dir.mkdir(parents=True, exist_ok=True)
        warmup = [_iteration(iterator, weights, count_bytes=i == 0) for i in range(cli.warmup)]
        torch.cuda.reset_peak_memory_stats()
        timings = [_iteration(iterator, weights) for _ in range(cli.iterations)]
        peak_bytes = [None] * dist.get_world_size()
        dist.all_gather_object(peak_bytes, torch.cuda.max_memory_allocated(), group=get_gloo_group())
        diagnostic = _profile_iteration(cli, iterator, weights) if cli.profile or cli.torch_trace else None
        if dist.get_rank() == 0:
            result = {
                "label": cli.label,
                "scope": "standalone routed-expert iterator; synthetic weights; no training, serving, or transport",
                "source": _provenance(repo_root, model_script),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "ep": cli.ep_size,
                "edp": cli.edp_size,
                "tp": 4,
                "etp": 1,
                "pp": 1,
                "materialize_ranks": [0],
                "layers": layers,
                "experts": args.num_experts,
                "hidden": args.hidden_size,
                "intermediate": intermediate,
                "buffer_bytes": cli.buffer_bytes,
                "input_bytes_per_rank": sum(tensor.nbytes for tensor in weights.values()),
                "peak_cuda_allocated_bytes_per_rank": peak_bytes,
                "seed": cli.seed,
                "environment": NVFP4_ENV,
                "warmup": warmup,
                "timed": timings,
                "median_max_rank_seconds": statistics.median(row["max_rank_seconds"] for row in timings),
                "profile": diagnostic,
            }
            (cli.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print("EXPERT_ITERATOR_BENCHMARK " + json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
