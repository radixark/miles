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

import numpy as np
import torch
import torch.distributed as dist

ARMS = (("cpu-zstd", "cpu", "zstd"), ("gpu-zstd", "gpu", "zstd"), ("gpu-snappy", "gpu", "snappy"))
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
    parser.add_argument("--perturb-fraction", type=float, default=0.001, help="Approximate fraction of matrix elements selected")
    parser.add_argument("--perturb-relative-scale", type=float, default=0.03125, help="Selected weights multiply by 1 + this value")
    parser.add_argument("--timing", action="store_true", help="Record optional CUDA phase events; changes instrumentation overhead")
    args = parser.parse_args()
    if args.versions < 1 or not 0 < args.perturb_fraction <= 1 or not 0 < args.perturb_relative_scale < 1:
        parser.error("versions must be positive, fraction in (0, 1], and relative scale in (0, 1)")
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        parser.error("Launch with torchrun --standalone --nproc-per-node=8")
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
    os.environ["WEIGHT_DELTA_TIMING"] = str(int(args.timing))
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
    from miles.utils.external_utils.model_args_utils import load_model_args
    from tools.convert_hf_to_torch_dist import get_args

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
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "8",
        "--expert-tensor-parallel-size",
        "1",
        "--no-load-optim",
        "--no-load-rng",
        "--finetune",
    ]
    original_argv, sys.argv = sys.argv, argv
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    args.sglang_speculative_algorithm = "EAGLE"  # Static draft is excluded from target export.
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
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

    class TimedIterator(HfWeightIteratorDirect):
        def reset_timing(self):
            self.conversion_wall_s = 0.0
            self.conversion_events = []
            self.converted_units = 0

        def _convert_to_hf_param_units(self, named_params):
            iterator = super()._convert_to_hf_param_units(named_params)
            while True:
                started = time.monotonic()
                start = torch.cuda.Event(enable_timing=True) if timing else None
                if start is not None:
                    start.record()
                try:
                    unit = next(iterator)
                except StopIteration:
                    return
                if start is not None:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    self.conversion_events.append((start, end))
                self.conversion_wall_s += time.monotonic() - started
                self.converted_units += 1
                yield unit

    iterator = TimedIterator.build(
        args,
        model,
        required_placement=WeightUpdatePlacement(gather_pp=False),
        model_name=config["architectures"][0],
        quantization_config=config["quantization_config"],
    )
    iterator.reset_timing()
    return iterator


def _discover_plan(args, iterator, weights):
    from miles.backends.training_utils.parallel import get_parallel_state
    from miles.utils import disk_delta

    local, error = {}, None
    parallel = get_parallel_state()

    def consume(unit):
        nonlocal error
        try:
            for name, tensor in unit:
                if name in local:
                    raise ValueError(f"Repeated owner for {name}")
                dtype, shape = disk_delta.checkpoint_tensor_layout(args.hf_checkpoint, name)
                if tuple(tensor.shape) != shape:
                    raise ValueError(f"Exporter/checkpoint shape mismatch for {name}")
                expert = re.search(r"\.mlp\.experts\.(\d+)\.", name)
                if expert and int(expert[1]) // (args.num_experts // parallel.ep.size) != parallel.ep.rank:
                    raise ValueError(f"Nonlocal expert ownership for {name}")
                if not expert and dist.get_rank() != 0:
                    raise ValueError(f"Nonrouted tensor owner must be global rank 0: {name}")
                local[name] = {
                    "name": name,
                    "dtype": dtype,
                    "shape": list(shape),
                    "encoding": "replace_bytes" if ".self_attn.indexer.k_norm." in name else "xor_bytes",
                    "views": [{"id": "canonical", "slices": [[0, size] for size in shape]}],
                }
        except Exception as caught:
            error = error or caught
        return []

    iterator.set_local_expert_transform(prefetch=lambda _: None, transform=lambda _key, unit: consume(unit))
    for bucket in iterator.iter_hf_weights(weights, materialize=dist.get_rank() == 0):
        if dist.get_rank() == 0:
            consume(bucket)
    _check(error, "mutable inventory discovery")
    shards = _gather(list(local.values()))
    names = [tensor["name"] for shard in shards for tensor in shard]
    if len(set(names)) != len(names):
        raise ValueError("The real exporter did not assign exactly one owner per mutable tensor")
    plan = sorted([tensor for shard in shards for tensor in shard], key=lambda tensor: tensor["name"])
    return plan, {
        "rank": dist.get_rank(),
        "ep_rank": parallel.ep.rank,
        "edp_rank": parallel.edp.rank,
        "tensor_count": len(local),
        "routed_tensor_count": sum(".mlp.experts." in name for name in local),
        "names": sorted(local),
    }


def _make_protocol(args, plan, arm, output):
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
                            "identity": {"rank_id": "producer-benchmark-plan"},
                            "plan": {"tensors": plan},
                        }
                    ],
                }
            ]

        def _declare_baseline(self):
            pass  # Publication-only benchmark never sends receiver metadata RPCs.

        def acknowledge_publication(self):
            # The harness verifies the sealed publication before this call.
            # Production clears this only after successful receiver activation.
            self._uncommitted = False

    name, encoder, codec = arm
    os.environ["WEIGHT_DELTA_ENCODER"] = encoder
    os.environ["WEIGHT_DELTA_CODEC"] = codec
    arm_args = copy.copy(args)
    arm_args.update_weight_disk_dir = str(output / name / "publications")
    return ProducerOnlyProtocol(arm_args)


def _setup_protocols(args, plan, iterator, weights, output):
    from miles.backends.training_utils.parallel import get_parallel_state

    protocols, setup = {}, []
    for arm in ARMS:
        started = time.monotonic()
        protocol = _make_protocol(args, plan, arm, output)
        protocol.connect([], [], [], get_parallel_state(), iterator.placement, "target")
        if protocol.is_sender != (dist.get_rank() == 0):
            raise ValueError("EP8/TP1/PP1/CP1 requires rank 0 as the ordinary tensor sender")
        protocol.bind_iterator(iterator)
        initialized = protocol.begin_sync(0, lambda **kw: iterator.iter_hf_weights(weights, **kw))
        if initialized:
            raise RuntimeError("Expected baseline capture, not an update")
        protocols[arm[0]] = protocol
        setup.append({"arm": arm[0], "baseline_capture_s": time.monotonic() - started})
    return protocols, setup


def _perturb(weights, *, fraction, relative_scale, version):
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


def _verify_publication(publication, plan):
    path = Path(publication["manifest_path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != publication["manifest_sha256"]:
        raise ValueError("Publication manifest checksum mismatch")
    manifest = json.loads(raw)
    if {tensor["name"] for tensor in manifest["tensors"]} != {tensor["name"] for tensor in plan}:
        raise ValueError("Sealed publication does not cover the exact mutable exporter inventory")
    for item in manifest["files"]:
        if (path.parent / item["name"]).stat().st_size != item["nbytes"]:
            raise ValueError("Sealed publication payload size mismatch")
    return {
        "manifest_bytes": len(raw),
        "payload_bytes": sum(item["nbytes"] for item in manifest["files"]),
        "canonical_bytes": sum(tensor["nbytes"] for tensor in manifest["tensors"]),
        "changed_bytes": sum(tensor["changed_bytes"] for tensor in manifest["tensors"]),
        "tensor_count": len(manifest["tensors"]),
    }


def _run_arm(protocol, iterator, weights, version, plan):
    protocol.bind_iterator(iterator)
    iterator.reset_timing()
    dist.barrier()
    torch.cuda.synchronize()
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
    # Completion only, once per arm; no per-conversion timing synchronizations.
    torch.cuda.synchronize()
    completed = time.monotonic()
    conversion_cuda_ms = sum(start.elapsed_time(end) for start, end in iterator.conversion_events) if iterator.conversion_events else None
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
    }
    error, sizes = None, None
    if dist.get_rank() == 0:
        try:
            sizes = _verify_publication(publication, plan)
        except Exception as caught:
            error = caught
    _check(error, "sealed publication validation")
    protocol.acknowledge_publication()
    return {"ranks": _gather(measurement), "publication": publication, "sizes": sizes}


def _verify_equal_targets(protocols):
    snapshots = [protocol._snapshot for protocol in protocols.values()]
    if any(snapshot.keys() != snapshots[0].keys() for snapshot in snapshots[1:]):
        raise ValueError("Canonical ownership differs between benchmark arms")
    byte_count = 0
    for name, baseline in snapshots[0].items():
        baseline = baseline.numpy() if isinstance(baseline, torch.Tensor) else baseline
        for snapshot in snapshots[1:]:
            target = snapshot[name]
            target = target.numpy() if isinstance(target, torch.Tensor) else target
            if not np.array_equal(baseline, target):
                raise ValueError(f"Quantized targets differ between arms for {name}")
        byte_count += baseline.nbytes
    return {"rank": dist.get_rank(), "equal": True, "canonical_bytes": byte_count, "tensor_count": len(snapshots[0])}


def _versions(options, protocols, iterator, weights, plan):
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
        # Rotate order so a single arm does not always see the first export.
        order = list(protocols)
        offset = (version - 1) % len(order)
        order = order[offset:] + order[:offset]
        arms = {}
        for name in order:
            arms[name] = _run_arm(protocols[name], iterator, weights, version, plan)
            _write_root(options.output / f"version-{version:03d}-{name}.json", arms[name])
        error, equality = None, None
        try:
            equality = _verify_equal_targets(protocols)
        except Exception as caught:
            error = caught
        _check(error, "same quantized target comparison outside measured intervals")
        result = {"version": version, "order": order, "perturbation": perturbation, "arms": arms, "equality": _gather(equality)}
        results.append(result)
        _write_root(options.output / f"version-{version:03d}.json", result)
        if dist.get_rank() == 0:
            print(json.dumps({"version": version, "same_targets": True, "arms": {name: {"sizes": value["sizes"], "blocked_s": [rank["producer_blocked_s"] for rank in value["ranks"]]} for name, value in arms.items()}}), flush=True)
    return results


def _runtime_metadata():
    versions = {}
    for package in ("torch", "transformer-engine", "flashinfer-python", "nvidia-libnvcomp-cu13"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"hostname": socket.gethostname(), "pid": os.getpid(), "rank": dist.get_rank(), "device": torch.cuda.get_device_name(), "versions": versions}


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
    protocols, baseline_setup = _setup_protocols(args, plan, iterator, weights, options.output)
    runtime, error = None, None
    try:
        runtime = _runtime_metadata()
    except Exception as caught:
        error = caught
    _check(error, "runtime metadata")
    setup = _gather({"load_s": load_s, "discovery_s": discovery_s, "baseline": baseline_setup, "ownership": ownership, "runtime": runtime})
    _write_root(options.output / "plan.json", {"scope": "producer-only canonical full views", "tensors": plan})
    _write_root(
        options.output / "setup.json",
        {
            "model_args": model_argv,
            "env": NVFP4_ENV,
            "timing": options.timing,
            "ranks": setup,
            "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
            "source_digest": os.environ.get("GPU_DELTA_SOURCE_DIGEST"),
        },
    )
    results = _versions(options, protocols, iterator, weights, plan)
    _write_root(options.output / "result.json", {"success": True, "scope": "producer-only; no receiver or optimizer update", "versions": results})


def main():
    options = parse_args()
    try:
        run(options)
    except BaseException as error:
        if options.output.is_dir():
            _write_json(options.output / f"failure-rank-{os.environ.get('RANK', 'unknown')}.json", {"success": False, "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
