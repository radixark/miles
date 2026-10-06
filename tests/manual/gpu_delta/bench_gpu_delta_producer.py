"""One-node EP8 producer benchmark using the real GLM-5.2 export/publication path.

No SGLang engines, training forward/backward, or receiver activation are involved.
See bench_gpu_delta_producer.md for setup, interpretation, and limitations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

NVFP4_ENV = {
    "SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN": "0",
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
    "INDEXER_ROPE_NEOX_STYLE": "0",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", type=Path, required=True, help="Prepared five-layer NVFP4 HF checkpoint")
    parser.add_argument("--load", type=Path, required=True, help="Matching native-DSA Megatron torch_dist checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is rejected")
    parser.add_argument("--versions", type=int, default=3)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument(
        "--perturb-fraction", type=float, default=0.001, help="Approximate fraction of matrix elements selected"
    )
    parser.add_argument(
        "--perturb-relative-scale", type=float, default=0.03125, help="Selected weights multiply by 1 + this value"
    )
    parser.add_argument(
        "--timing", action="store_true", help="Record optional CUDA phase events; changes instrumentation overhead"
    )
    args = parser.parse_args()
    if args.versions < 1 or not 0 < args.perturb_fraction <= 1 or not 0 < args.perturb_relative_scale < 1:
        parser.error("versions must be positive, fraction in (0, 1], and relative scale in (0, 1)")
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        parser.error("Launch with torchrun --standalone --nproc-per-node=8")
    if (
        args.tensor_model_parallel_size < 1
        or args.context_parallel_size < 1
        or 8 % (args.tensor_model_parallel_size * args.context_parallel_size)
    ):
        parser.error("TP and CP must be positive and TP × CP must divide eight ranks")
    return args


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _gather(value):
    from miles.utils.distributed_utils import get_gloo_group

    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value, group=get_gloo_group())
    return values


def _check(error, phase):
    errors = _gather(None if error is None else f"{type(error).__name__}: {error}")
    if any(errors):
        raise RuntimeError(f"Producer benchmark {phase} failed: {errors}") from error


def _write_root(path, value):
    error = None
    if dist.get_rank() == 0:
        try:
            _write_json(path, value)
        except Exception as caught:
            error = caught
    _check(error, f"writing {path.name}")


def _environment(args):
    for key, value in NVFP4_ENV.items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError(f"Benchmark requires {key}={value}, received {os.environ[key]!r}")
        os.environ[key] = value
    os.environ["GPU_DELTA_TIMING"] = str(int(args.timing))
    config = json.loads((args.hf_checkpoint / "config.json").read_text())
    if config.get("model_type") != "glm_moe_dsa" or config.get("num_hidden_layers") != 5:
        raise ValueError("Expected the native GLM-5.2 five-layer checkpoint")
    quantization = config.get("quantization_config", {})
    if quantization.get("quant_algo") != "NVFP4" and quantization.get("quant_method") != "nvfp4":
        raise ValueError("Expected an NVFP4 checkpoint")
    return config


def _model_args(options):
    # Reuse the maintained offline conversion parser/model declaration. Imports
    # are deferred so --help works without Megatron, TE, or a GPU installation.
    from tools.convert_hf_to_torch_dist import get_args

    from miles.utils.external_utils.model_args_utils import load_model_args

    argv = [
        "bench_gpu_delta_producer",
        *shlex.split(load_model_args("glm5.2-744B-A40B_5layer")),
        "--hf-checkpoint",
        str(options.hf_checkpoint),
        "--load",
        str(options.load),
        "--megatron-to-hf-mode",
        "raw",
        "--dsa-impl",
        "megatron",
        "--dsa-kernel-backend",
        "cudnn",
        "--cp-comm-type",
        "allgather",
        "--tensor-model-parallel-size",
        str(options.tensor_model_parallel_size),
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        str(options.context_parallel_size),
        "--expert-model-parallel-size",
        "8",
        "--expert-tensor-parallel-size",
        "1",
        "--no-load-optim",
        "--no-load-rng",
        "--finetune",
    ]
    if options.tensor_model_parallel_size > 1:
        # Native AbsorbedMLA requires sequence parallelism with tensor parallelism.
        argv.append("--sequence-parallel")
    original_argv, sys.argv = sys.argv, argv
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    args.sglang_speculative_algorithm = "EAGLE"  # Static draft is excluded from target export.
    args.update_weight_transfer_mode = "gpu-delta"
    args.update_weight_delta_initial_sync = False
    args.update_weight_buffer_size = 512 * 1024**2
    args.extra_high_precision_layers_megatron = (".shared_experts.linear_fc1", ".shared_experts.linear_fc2")
    args.custom_update_weight_post_write_path = None
    return args, argv[1:]


def _load_model(args):
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model

    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
    from miles.backends.megatron_utils.initialize import init
    from miles.backends.megatron_utils.model_provider import get_model_provider_func
    from miles.backends.megatron_utils.named_weights import named_params_and_buffers

    init(args)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    enforce_marked_param_dtypes(model)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    weights = dict(named_params_and_buffers(args, model))
    torch.cuda.synchronize()
    return model, weights


def _make_iterator(args, model, config, timing):
    from miles.backends.megatron_utils.update_weight.gpu_delta_export import HfWeightIteratorGpuDelta
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

    class TimedIterator(HfWeightIteratorGpuDelta):
        def reset_timing(self):
            self.conversion_wall_s = 0.0
            self.conversion_events = []
            self.converted_units = 0

        def _convert_to_hf_param_units(self, named_params):
            iterator = super()._convert_to_hf_param_units(named_params)
            for native_name, _ in named_params:
                started = time.monotonic()
                start = torch.cuda.Event(enable_timing=True) if timing else None
                if start is not None:
                    start.record()
                unit = next(iterator)
                if start is not None:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    self.conversion_events.append((start, end))
                self.conversion_wall_s += time.monotonic() - started
                self.converted_units += 1
                if self.discovery_units is not None:
                    if native_name in self.discovery_units:
                        raise ValueError(f"Repeated native conversion for {native_name}")
                    self.discovery_units[native_name] = [name for name, _ in unit]
                yield unit

    iterator = TimedIterator.build(
        args,
        model,
        required_placement=WeightUpdatePlacement(gather_pp=False),
        model_name=config["architectures"][0],
        quantization_config=config["quantization_config"],
    )
    iterator.reset_timing()
    iterator.discovery_units = None
    return iterator


def _discover_plan(args, iterator, weights):
    from miles.backends.training_utils.parallel import get_parallel_state
    from miles.utils.gpu_delta_publication import checkpoint_tensor_layout

    local, error = {}, None
    parallel = get_parallel_state()

    def record_error(caught):
        nonlocal error
        error = error or caught

    def consume(unit):
        try:
            for name, tensor in unit:
                if name in local:
                    raise ValueError(f"Repeated owner for {name}")
                dtype, shape = checkpoint_tensor_layout(args.hf_checkpoint, name)
                if tuple(tensor.shape) != shape:
                    raise ValueError(f"Exporter/checkpoint shape mismatch for {name}")
                expert = re.search(r"\.mlp\.experts\.(\d+)\.", name)
                if expert and int(expert[1]) // (args.num_experts // parallel.ep.size) != parallel.ep.rank:
                    raise ValueError(f"Nonlocal expert ownership for {name}")
                local[name] = {
                    "name": name,
                    "dtype": dtype,
                    "shape": list(shape),
                    "encoding": "raw_bytes" if len(shape) <= 1 else "xor_bytes",
                    "views": [{"id": "canonical", "slices": [[0, size] for size in shape]}],
                }
        except Exception as caught:
            record_error(caught)

    iterator.local_consumer = consume
    iterator.local_error_consumer = record_error
    iterator.discovery_units = {}
    for bucket in iterator.iter_hf_weights(weights, materialize=dist.get_rank() == 0):
        if dist.get_rank() == 0:
            consume(bucket)
    _check(error, "mutable inventory discovery")
    converted = _gather(iterator.discovery_units)
    iterator.discovery_units = None
    ordinary = {}
    for rank, units in enumerate(converted):
        for name in units:
            if name in iterator.ordinary_owners:
                if name in ordinary or iterator.ordinary_owners[name] != rank:
                    raise ValueError(f"Conversion differs from the fixed ordinary owner plan: {name}")
                ordinary[name] = rank
    if ordinary != iterator.ordinary_owners:
        raise ValueError("Discovery omitted a native tensor from the ordinary owner plan")
    shards = _gather(list(local.values()))
    names = [tensor["name"] for shard in shards for tensor in shard]
    if len(set(names)) != len(names):
        raise ValueError("The real exporter did not assign exactly one owner per mutable tensor")
    converted_names = [name for units in converted for outputs in units.values() for name in outputs]
    if sorted(converted_names) != sorted(names):
        raise ValueError("Consumed canonical inventory differs from the complete converted units")
    plan = sorted([tensor for shard in shards for tensor in shard], key=lambda tensor: tensor["name"])
    return plan, {
        "rank": dist.get_rank(),
        "ep_rank": parallel.ep.rank,
        "edp_rank": parallel.edp.rank,
        "tp_rank": parallel.tp.rank,
        "cp_rank": parallel.cp.rank,
        "dp_rank": parallel.intra_dp.rank,
        "native_units": converted[dist.get_rank()],
        "ordinary_owners": dict(iterator.ordinary_owners),
        "tensor_count": len(local),
        "routed_tensor_count": sum(".mlp.experts." in name for name in local),
        "names": sorted(local),
    }


def _make_protocol(args, plan, output):
    from miles.backends.training_utils.weight_update.protocols.gpu_delta import UpdateWeightFromGpuDelta

    class ProducerOnlyProtocol(UpdateWeightFromGpuDelta):
        async def _describe(self):
            # Explicitly constructed exporter metadata, NOT a receiver identity
            # or capability proof. There is no receiver in this benchmark.
            return [
                {
                    "success": True,
                    "participants": [
                        {
                            "identity": {
                                "rank_id": "producer-benchmark-plan",
                                "engine_id": "producer-benchmark-no-receiver",
                                "host_cache_id": "producer-benchmark-no-host-cache",
                            },
                            "plan": {"codec": self.codec, "tensors": plan},
                        }
                    ],
                }
            ]

        def _declare_baseline(self):
            pass  # Publication-only benchmark never sends receiver metadata RPCs.

    protocol_args = copy.copy(args)
    protocol_args.update_weight_disk_dir = str(output / "publications")
    return ProducerOnlyProtocol(protocol_args)


def _setup_protocol(args, plan, iterator, weights, output):
    from miles.backends.training_utils.parallel import get_parallel_state

    started = time.monotonic()
    protocol = _make_protocol(args, plan, output)
    protocol.connect([], [], [], get_parallel_state(), iterator.placement, "target")
    iterator.local_consumer = protocol.send_bucket
    iterator.local_error_consumer = protocol.record_export_error
    if protocol.begin_sync(0, lambda **kw: iterator.iter_hf_weights(weights, **kw)):
        raise RuntimeError("Expected baseline capture, not an update")
    return protocol, {"baseline_capture_s": time.monotonic() - started}


def _perturb(weights, fraction, relative_scale, version):
    stride = max(1, round(1 / fraction))
    selected, eligible = 0, 0
    with torch.no_grad():
        for name, tensor in sorted(weights.items()):
            if not tensor.is_floating_point() or tensor.ndim < 2:
                continue
            if not tensor.is_contiguous():
                raise ValueError(f"Perturbation requires a contiguous training matrix: {name}")
            offset = int.from_bytes(hashlib.sha256(f"{name}:{version}".encode()).digest()[:8], "little") % stride
            view = tensor.view(-1)[offset::stride]
            view.mul_(1 + relative_scale)
            selected += view.numel()
            eligible += tensor.numel()
    torch.cuda.synchronize()
    return {"selected_elements": selected, "eligible_elements": eligible, "stride": stride}


def _verify_publication(publication, plan, codec):
    path = Path(publication["manifest_path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != publication["manifest_sha256"]:
        raise ValueError("Publication manifest checksum mismatch")
    manifest = json.loads(raw)
    if (
        manifest.get("codec") != codec
        or manifest.get("protocol_version") != 4
        or manifest.get("frame_bytes") != 1 << 20
    ):
        raise ValueError(f"Sealed publication must use protocol 4 / {codec}")
    if {tensor["name"] for tensor in manifest["tensors"]} != {tensor["name"] for tensor in plan}:
        raise ValueError("Sealed publication does not cover the exact mutable exporter inventory")
    for tensor in manifest["tensors"]:
        is_raw = len(tensor["shape"]) <= 1
        if tensor["encoding"] != ("raw_bytes" if is_raw else "xor_bytes"):
            raise ValueError("Sealed publication differs from the shape-based direct-value codec")
        if is_raw and (
            tensor["frames"]
            or "outer" in tensor
            or tensor.get("raw", {}).get("encoded_bytes", 0) != (tensor["nbytes"] if tensor["changed_bytes"] else 0)
        ):
            raise ValueError("Scalar/vector must transfer its complete target without compression")
    for item in manifest["files"]:
        if (path.parent / item["name"]).stat().st_size != item["nbytes"]:
            raise ValueError("Sealed publication payload size mismatch")
    return {
        "manifest_bytes": len(raw),
        "payload_bytes": sum(item["nbytes"] for item in manifest["files"]),
        "canonical_bytes": sum(tensor["nbytes"] for tensor in manifest["tensors"]),
        "changed_bytes": sum(tensor["changed_bytes"] for tensor in manifest["tensors"]),
        "tensor_count": len(manifest["tensors"]),
        "codec": manifest["codec"],
        "frame_bytes": 1 << 20,
        "raw_tensor_count": sum(tensor["encoding"] == "raw_bytes" for tensor in manifest["tensors"]),
        "raw_changed_tensors": sum("raw" in tensor for tensor in manifest["tensors"]),
        "raw_bytes": sum(tensor.get("raw", {}).get("encoded_bytes", 0) for tensor in manifest["tensors"]),
        "inner_encoded_frame_bytes": sum(
            frame["encoded_bytes"] for tensor in manifest["tensors"] for frame in tensor["frames"]
        ),
        "outer_stored_bytes": sum(tensor.get("outer", {}).get("encoded_bytes", 0) for tensor in manifest["tensors"]),
        "outer_decoded_arena_bytes": sum(
            tensor.get("outer", {}).get("decoded_bytes", 0) for tensor in manifest["tensors"]
        ),
    }


def _run_update(protocol, iterator, weights, version, plan):
    iterator.reset_timing()
    dist.barrier()
    torch.cuda.synchronize()
    # The existing isolated benchmark fence brackets allocator accounting; no
    # additional CUDA fence or production allocator policy is introduced.
    memory_before = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if not protocol.begin_sync(version, lambda **kw: iterator.iter_hf_weights(weights, **kw)):
        raise RuntimeError("Measured update unexpectedly performed baseline capture")
    setup_end = time.monotonic()
    for bucket in iterator.iter_hf_weights(weights, materialize=protocol.is_sender):
        if protocol.is_sender:
            protocol.send_bucket(bucket)
    export_end = time.monotonic()
    protocol.after_base_weights()
    tail_end = time.monotonic()
    publication = protocol.publish(version)
    sealed = time.monotonic()
    # Completion only, once per update; no per-conversion timing synchronizations.
    torch.cuda.synchronize()
    completed = time.monotonic()
    conversion_cuda_ms = (
        sum(start.elapsed_time(end) for start, end in iterator.conversion_events)
        if iterator.conversion_events
        else None
    )
    measurement = {
        "rank": dist.get_rank(),
        "setup_s": setup_end - started,
        "export_loop_s": export_end - setup_end,
        "encoding_tail_s": tail_end - export_end,
        "seal_and_visibility_s": sealed - tail_end,
        "completion_fence_s": completed - sealed,
        "producer_blocked_s": completed - started,
        "conversion_host_s": iterator.conversion_wall_s,
        "conversion_cuda_ms": conversion_cuda_ms,
        "converted_units": iterator.converted_units,
        "conversion_event_count": 2 * len(iterator.conversion_events),
        "gpu_memory_bytes": {
            "before_allocated": memory_before["allocated"],
            "before_reserved": memory_before["reserved"],
            "after_allocated": torch.cuda.memory_allocated(),
            "after_reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        },
    }
    error, sizes = None, None
    if dist.get_rank() == 0:
        try:
            sizes = _verify_publication(publication, plan, protocol.codec)
        except Exception as caught:
            error = caught
    _check(error, "sealed publication validation")
    ranks = _gather(measurement)
    return {
        "version": version,
        "measurement_phase": "first-use-allocation" if version == 1 else "warm-update",
        "ranks": ranks,
        "rank_max_s": {key: max(rank[key] for rank in ranks) for key in measurement if key.endswith("_s")},
        "publication": publication,
        "sizes": sizes,
    }


def _verify_pending_inventory(protocol, owned_plan):
    # This selected-codec run has no paired control. Check the complete owned pending
    # target inventory before simulating a successful receiver acknowledgment.
    from math import prod

    from miles.utils.gpu_delta_publication import DTYPE_BYTES

    pending = protocol.pending_baseline
    if pending.keys() != owned_plan.keys():
        raise ValueError("Pending canonical ownership differs from the committed baseline")
    byte_count = 0
    for name, target in pending.items():
        spec = owned_plan[name]
        if target.numel() * target.element_size() != prod(spec["shape"]) * DTYPE_BYTES[spec["dtype"]]:
            raise ValueError(f"Pending canonical size differs for {name}")
        byte_count += target.numel() * target.element_size()
    return {"rank": dist.get_rank(), "canonical_bytes": byte_count, "tensor_count": len(pending)}


def _versions(options, protocol, iterator, weights, plan, owned_plan):
    results = []
    for version in range(1, options.versions + 1):
        perturbation = _gather(
            _perturb(
                weights,
                fraction=options.perturb_fraction,
                relative_scale=options.perturb_relative_scale,
                version=version,
            )
        )
        measurement = _run_update(protocol, iterator, weights, version, plan)
        error, inventory = None, None
        try:
            inventory = _verify_pending_inventory(protocol, owned_plan)
        except Exception as caught:
            error = caught
        _check(error, "pending quantized target inventory outside measured interval")
        # Production commits only after successful receiver activation. This
        # isolated producer benchmark explicitly simulates that acknowledgment.
        error = None
        try:
            protocol.commit_pending_baseline()
        except Exception as caught:
            error = caught
        _check(error, "producer-only baseline commit")
        result = {
            "version": version,
            "codec": protocol.codec,
            "perturbation": perturbation,
            "measurement_phase": "first-use-allocation" if version == 1 else "warm-update",
            "measurement": measurement,
            "inventory": _gather(inventory),
            "baseline_commit": "producer-only-after-sealing-and-inventory-check",
            "target_comparison": "not-performed-single-codec",
        }
        results.append(result)
        _write_root(options.output / f"version-{version:03d}.json", result)
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "version": version,
                        "codec": protocol.codec,
                        "sizes": measurement["sizes"],
                        "blocked_s": [rank["producer_blocked_s"] for rank in measurement["ranks"]],
                    }
                ),
                flush=True,
            )
    return results


def _runtime_metadata():
    versions = {}
    for package in (
        "torch",
        "transformer-engine",
        "flashinfer-python",
        "nvidia-libnvcomp-cu13",
        "zstandard",
        "python-snappy",
        "lz4",
        "cramjam",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": dist.get_rank(),
        "device": torch.cuda.get_device_name(),
        "versions": versions,
    }


def run(options):
    from miles.utils.distributed_utils import init_gloo_group

    config = _environment(options)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    init_gloo_group()
    # Initialization errors still retain torchrun's nonzero exit and per-rank log.
    args, model_argv = _model_args(options)
    load_started = time.monotonic()
    model, weights = _load_model(args)
    load_s = time.monotonic() - load_started
    error = None
    if dist.get_rank() == 0:
        try:
            options.output.mkdir(parents=True, exist_ok=False)
        except Exception as caught:
            error = caught
    _check(error, "new output directory")
    iterator = _make_iterator(args, model, config, options.timing)
    discover_started = time.monotonic()
    plan, ownership = _discover_plan(args, iterator, weights)
    discovery_s = time.monotonic() - discover_started
    protocol, baseline_setup = _setup_protocol(args, plan, iterator, weights, options.output)
    runtime, error = None, None
    try:
        runtime = _runtime_metadata()
    except Exception as caught:
        error = caught
    _check(error, "runtime metadata")
    setup = _gather(
        {
            "load_s": load_s,
            "discovery_s": discovery_s,
            "baseline": baseline_setup,
            "ownership": ownership,
            "runtime": runtime,
        }
    )
    _write_root(options.output / "plan.json", {"scope": "producer-only canonical full views", "tensors": plan})
    _write_root(
        options.output / "setup.json",
        {
            "model_args": model_argv,
            "env": NVFP4_ENV,
            "timing": options.timing,
            "codec": protocol.codec,
            "frame_bytes": 1 << 20,
            "producer_pipeline": f"pinned-snapshot-bulk-gpu-{protocol.codec.removesuffix('-zstd')}-then-owner-wide-gpu-zstd",
            "gpu_batch_target_bytes": args.update_weight_buffer_size,
            "baseline_commit_scope": "producer-only-simulated-activation-after-inventory-check",
            "ranks": setup,
            "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
            "source_digest": os.environ.get("GPU_DELTA_SOURCE_DIGEST"),
            "timing_interpretation": "Version 1 includes first-use allocation/compilation; compare warm versions 2/3 separately. Cumulative versions are not independent fixed-target repetitions; this single-codec run does not prove target equality with a previous run.",
        },
    )
    owned_plan = {tensor["name"]: tensor for tensor in plan if tensor["name"] in set(ownership["names"])}
    results = _versions(options, protocol, iterator, weights, plan, owned_plan)
    _write_root(
        options.output / "result.json",
        {"success": True, "scope": "producer-only; no receiver or optimizer update", "versions": results},
    )


def main():
    options = parse_args()
    try:
        run(options)
    except BaseException as error:
        if options.output.is_dir():
            _write_json(
                options.output / f"failure-rank-{os.environ.get('RANK', 'unknown')}.json",
                {"success": False, "error": f"{type(error).__name__}: {error}"},
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
