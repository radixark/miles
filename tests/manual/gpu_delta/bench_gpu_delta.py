"""Benchmark GPU-delta compression and receiver updates from one NVFP4 HF base.

Run from the Miles checkout with paired SGLang on PYTHONPATH::

    python tests/manual/gpu_delta/bench_gpu_delta.py --model /models/base

Requires eight Blackwell GPUs, CUDA 13 and nvCOMP >=5.3. The GLM5.2 W4A16
workload uses static bundled MTP and two independent TP4/DP4/EP4 receivers.
Sender processes split whole layers across GPUs; no Megatron checkpoint is needed.
Defaults: three codecs x payload hashing on/off, all with 1 MiB inner frames.
Use --reuse-fixtures with a reviewed producer SPEC to skip encoding and apply
saved deltas. Reused sender timings are historical; receiver timings are fresh.
The expensive second model load/oracle is opt-in with --validate-final-target.
Without it, reports explicitly remain NOT_VALIDATED for target equivalence.
Use --profile-version N with saved fixtures under an external Nsight Systems
launcher to gate capture around one warm receiver prepare/apply/resume update.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import _fixture_reuse
import httpx
import numpy as np
import zstandard

from miles.backends.training_utils.weight_update.protocols.gpu_delta.session import (
    activate_publication,
    merge_plans,
    negotiate_cohort,
)
from miles.utils.gpu_delta.publication import (
    CODECS,
    DTYPE_BYTES,
    FRAME_BYTES,
    PublicationWriter,
    canonical_json,
    configured_codec,
    configured_skip_payload_hash,
    seal_publication,
    sha256,
)

# Same rollout topology/precision/MTP as the GLM5.2 W4A16 recipe. CuTe DSL + no
# MoE A2A is deliberate: this public branch does not require MegaMoE integration.
SERVER_ARGS = {
    "dtype": "bfloat16",
    "kv_cache_dtype": "fp8_e4m3",
    "tp_size": 8,
    "dp_size": 8,
    "ep_size": 8,
    "pp_size": 1,
    "enable_dp_attention": True,
    "enable_dp_lm_head": True,
    "enable_fp32_lm_head": True,
    "attention_backend": "dsa",
    "dsa_prefill_backend": "flashmla_sparse",
    "dsa_decode_backend": "flashmla_kv",
    "dsa_topk_backend": "flashinfer",
    "moe_runner_backend": "flashinfer_cutedsl",
    "moe_a2a_backend": "none",
    "moe_dense_tp_size": 1,
    "disable_shared_experts_fusion": True,
    "mem_fraction_static": 0.83,
    "max_running_requests": 128,
    "page_size": 64,
    "chunked_prefill_size": 32768,
    "max_prefill_tokens": 32768,
    "cuda_graph_backend_prefill": "disabled",
    "disable_flashinfer_autotune": True,
    "cuda_graph_backend_decode": "full",
    "cuda_graph_bs_decode": [1],
    "cuda_graph_max_bs_decode": 1,
    "reasoning_parser": "glm45",
    "speculative_algorithm": "EAGLE",
    "speculative_num_steps": 3,
    "speculative_eagle_topk": 1,
    "speculative_num_draft_tokens": 4,
    "speculative_moe_runner_backend": "triton",
    "speculative_moe_a2a_backend": "none",
    "speculative_dsa_topk_backend": "sgl-kernel",
    "speculative_draft_model_quantization": "unquant",
    "enable_draft_weights_cpu_backup": True,
    "tokenizer_worker_num": 1,
    "weight_cache_mode": "off",
    "model_loader_extra_config": '{"enable_multithread_load":true,"num_threads":64}',
    "trust_remote_code": True,
    "skip_server_warmup": True,
}
SERVER_ENV = {
    "SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN": "0",
    "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "0",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
    "SGLANG_RUST_BUILD_MODE": "never",
    "SGLANG_BUILD_RUST_EXTS": "none",
    "PYTHONUNBUFFERED": "1",
}


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _engine_specs(ports):
    if len(ports) not in (1, 2) or len(set(ports)) != len(ports) or any(not 0 < port < 65536 for port in ports):
        raise ValueError("--ports requires one or two distinct valid TCP ports")
    size = 8 // len(ports)
    return [
        {
            "engine_id": f"engine-{i:05d}",
            "port": port,
            "parallel_size": size,
            "gpu_ids": ",".join(str(gpu) for gpu in range(i * size, (i + 1) * size)),
        }
        for i, port in enumerate(ports)
    ]


def _validate_cohort(cohort, engine_count):
    size = 8 // engine_count
    if sum(len(group) for group in cohort.participants) != 8 or len(cohort.participants) != engine_count:
        raise ValueError("Expected eight original participants across the requested engines")
    for engine, participants in enumerate(cohort.participants):
        if (
            len(participants) != size
            or cohort.engine_ids[engine] != f"engine-{engine:05d}"
            or {p["dp_rank"] for p in participants} != set(range(size))
            or {p["tp_rank"] for p in participants} != set(range(size))
            or any(p["pp_rank"] != 0 for p in participants)
        ):
            raise ValueError("Engine participant topology differs from the requested TP/DP/EP layout")
    arena_ids = [{p["host_cache_id"] for p in participants} for participants in cohort.participants]
    if any(len(ids) != 1 for ids in arena_ids) or len(set().union(*arena_ids)) != engine_count:
        raise ValueError("Each benchmark engine must own one distinct host arena")


def _capture_gpu_processes(cohort, ports):
    """Untimed native PID/UUID observation, separate from requested GPU masks."""
    device_text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    process_text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True
    )
    devices = {
        int(index.strip()): uuid.strip()
        for index, uuid in (line.split(",") for line in device_text.splitlines() if line.strip())
    }
    processes = [
        (int(pid.strip()), uuid.strip())
        for pid, uuid in (line.split(",") for line in process_text.splitlines() if line.strip())
    ]
    joined = []
    for spec, participants in zip(_engine_specs(ports), cohort.participants, strict=True):
        expected = {devices[int(index)] for index in spec["gpu_ids"].split(",")}
        actual = set()
        for participant in participants:
            candidates = _pid_candidates(participant["pid"])
            uuids = {uuid for pid, uuid in processes if pid in candidates}
            matched = len(uuids) == 1 and uuids <= expected
            if matched:
                actual.update(uuids)
            joined.append(
                {
                    "identity": participant,
                    "namespace_pids": candidates,
                    "gpu_uuid": next(iter(uuids)) if matched else None,
                    "candidate_gpu_uuids": sorted(uuids),
                }
            )
        if actual != expected:
            for row in joined:
                if row["identity"]["engine_id"] == spec["engine_id"]:
                    row["gpu_uuid"] = None
    # NVML can expose host PIDs outside the container's /proc namespace. Preserve
    # that observation instead of treating an unavailable join as a model error.
    return {
        "devices_raw": device_text,
        "compute_apps_raw": process_text,
        "participants": joined,
        "status": "MAPPED" if all(row["gpu_uuid"] for row in joined) else "UNQUALIFIED_PID_NAMESPACE",
    }


def _pid_candidates(pid):
    status = Path(f"/proc/{pid}/status").read_text()
    nested = next((line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")), [])
    return sorted({pid, *map(int, nested)})


def _tensor_index(model):
    index = {}
    for shard in sorted(model.glob("*.safetensors")):
        with shard.open("rb") as stream:
            header_bytes = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_bytes))
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in index:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            start, end = spec["data_offsets"]
            if start < 0 or end < start or end + 8 + header_bytes > shard.stat().st_size:
                raise ValueError(f"Invalid checkpoint tensor range: {name}")
            index[name] = spec | {"shard": shard.name, "offset": 8 + header_bytes + start, "nbytes": end - start}
    return index


def _mutate(raw, name, dtype, seed, version, rate):
    """Finite LSB perturbations; leave scale/routing metadata and static draft alone.

    Packed FP4 toggles a low-nibble value bit; BF16/FP32 toggles only the least
    significant mantissa byte. The explicit per-frame RNG avoids model-size RNG
    arrays. This synthetic distribution is not a claim about real RL deltas.
    """
    result = raw.copy()
    if not name.endswith(".weight") or dtype not in {"U8", "I8", "BF16", "F16", "F32"}:
        return result
    itemsize = DTYPE_BYTES[dtype]
    key = hashlib.sha256(f"{seed}:{version}:{name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    for start in range(0, result.size, FRAME_BYTES):
        count = min(FRAME_BYTES, result.size - start) // itemsize
        if count:
            n = max(1, round(rate * count))
            positions = np.unique(rng.integers(0, count, size=n)) * itemsize + start
            result[positions] ^= np.uint8(1)
    return result


def _calibrate(plan, seed, target_ratio):
    # Calibrate on representative frame geometries, including BF16 lane spacing.
    # Full publication overhead is measured after encoding, never silently forced
    # to the requested ratio by changing the Snappy target independently.
    compressor = zstandard.ZstdCompressor(level=1)
    samples = [t for t in plan if t["name"].endswith(".weight") and t["dtype"] in {"U8", "BF16", "F32"}]
    samples = samples[:: max(1, len(samples) // 24)][:24]
    if not samples:
        raise ValueError("No supported mutable weights in receiver plan")
    low, high = 0.000001, 0.02
    for _ in range(16):
        rate = (low + high) / 2
        total = encoded = 0
        for tensor in samples:
            raw = np.zeros(FRAME_BYTES, dtype=np.uint8)
            delta = _mutate(raw, name=tensor["name"], dtype=tensor["dtype"], seed=seed, version=1, rate=rate)
            encoded += len(compressor.compress(delta))
            total += raw.size
        if encoded / total < target_ratio:
            low = rate
        else:
            high = rate
    return (low + high) / 2


def _prepared_settings(args):
    files = []
    for path in sorted(args.model.iterdir()):
        if path.is_file() and path.suffix in {".json", ".py", ".model", ".tiktoken", ".safetensors"}:
            stat = path.stat()
            files.append({"path": path.name, "nbytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "model": str(args.model.resolve()),
        "base_files": files,
        "seed": args.seed,
        "ratio": args.ratio,
        "versions": args.versions,
        "mutation_sha256": sha256(inspect.getsource(_mutate).encode()),
        "mutation_frame_bytes": FRAME_BYTES,
    }


def _prepare_owner(args, owner, index, rate):
    """CPU-only preparation; owners write disjoint canonical ranges."""
    cache = args.prepared_inputs
    outputs = [(cache / "data" / f"v{version}.bin").open("r+b") for version in range(1, args.versions)]
    try:
        for tensor in owner["tensors"]:
            spec = index[tensor["name"]]
            with (args.model / spec["shard"]).open("rb") as source:
                source.seek(spec["offset"])
                current = np.frombuffer(source.read(spec["nbytes"]), dtype=np.uint8)
            if current.size != spec["nbytes"]:
                raise ValueError(f"Incomplete base tensor: {tensor['name']}")
            for version in range(1, args.versions + 1):
                current = _mutate(current, tensor["name"], spec["dtype"], args.seed, version, rate)
                if version < args.versions:
                    output = outputs[version - 1]
                    output.seek(spec["prepared_offset"])
                    output.write(current)
                else:
                    with (cache / "target-checkpoint" / spec["shard"]).open("r+b") as target:
                        target.seek(spec["offset"])
                        target.write(current)
    finally:
        for output in outputs:
            output.close()


def _prepare_inputs(args):
    """Materialize each cumulative target once, then publish READY last."""
    import torch

    started, cache = time.monotonic(), args.prepared_inputs
    settings = _prepared_settings(args)
    inventory_path = cache / "inventory" / "inventory.json"
    plan, digest = merge_plans(json.loads(inventory_path.read_text())["descriptions"])
    base_index, index, canonical_bytes = _tensor_index(args.model), {}, 0
    for tensor in plan:
        spec = base_index[tensor["name"]]
        if tensor["shape"] != spec["shape"] or tensor["dtype"] != spec["dtype"]:
            raise ValueError(f"Inventory/checkpoint mismatch: {tensor['name']}")
        index[tensor["name"]] = spec | {"prepared_offset": canonical_bytes}
        canonical_bytes += spec["nbytes"]
    data, target = cache / "data", cache / "target-checkpoint"
    if (cache / "READY.json").exists() or target.exists() or (data.exists() and any(data.iterdir())):
        raise ValueError(f"Prepared input cache already contains data; use a new directory: {cache}")
    data.mkdir(exist_ok=True)
    target.mkdir()
    for file in settings["base_files"]:
        shutil.copy2(args.model / file["path"], target / file["path"])
    index_path = data / "index.json"
    _save(index_path, index)
    raw_paths = [data / f"v{version}.bin" for version in range(1, args.versions)]
    for path in raw_paths:
        with path.open("xb") as output:
            output.truncate(canonical_bytes)
    rate = _calibrate(plan, args.seed, args.ratio)
    owner_count = torch.cuda.device_count() if args.sender_gpus is None else args.sender_gpus
    if owner_count < 1:
        raise ValueError("Input preparation requires at least one sender owner")
    owners = [owner for owner in _partition_plan(plan, owner_count) if owner["groups"]]
    if len(owners) == 1:
        _prepare_owner(args, owners[0], index, rate)
    else:
        with ProcessPoolExecutor(
            max_workers=len(owners), mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
        ) as pool:
            futures = [pool.submit(_prepare_owner, args, owner, index, rate) for owner in owners]
            for future in futures:
                future.result()
    paths = [inventory_path, index_path, *raw_paths, *sorted(target.iterdir())]
    for path in paths:
        with path.open("rb") as file:
            os.fsync(file.fileno())
    manifest = {
        "settings": settings,
        "rate": rate,
        "plan_digest": digest,
        "canonical_bytes": canonical_bytes,
        "files": [{"path": str(path.relative_to(cache)), "nbytes": path.stat().st_size} for path in paths],
        "preparation_wall_s": time.monotonic() - started,
    }
    pending = cache / "READY.json.pending"
    with pending.open("x") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.link(pending, cache / "READY.json")
    pending.unlink()
    return manifest


def _load_prepared_inputs(args):
    cache = args.prepared_inputs
    if not (cache / "READY.json").is_file():
        raise ValueError(f"Prepared input cache is incomplete; use a new directory: {cache}")
    manifest = json.loads((cache / "READY.json").read_text())
    if manifest["settings"] != _prepared_settings(args):
        raise ValueError("Prepared input settings differ from the requested immutable base or mutations")
    for file in manifest["files"]:
        path = cache / file["path"]
        if not path.is_file() or path.stat().st_size != file["nbytes"]:
            raise ValueError(f"Prepared input file is missing or has changed size: {path}")
    inventory = json.loads((cache / "inventory" / "inventory.json").read_text())
    _, digest = merge_plans(inventory["descriptions"])
    if digest != manifest["plan_digest"]:
        raise ValueError("Prepared input inventory plan differs from READY")
    return manifest


def _read_prepared_tensor(args, spec, version, pinned):
    if version == 0:
        path, offset = args.model / spec["shard"], spec["offset"]
    elif version == args.versions:
        path, offset = args.prepared_inputs / "target-checkpoint" / spec["shard"], spec["offset"]
    else:
        path, offset = args.prepared_inputs / "data" / f"v{version}.bin", spec["prepared_offset"]
    if pinned:
        import torch

        result = torch.empty(spec["nbytes"], dtype=torch.uint8, pin_memory=True)
        array = result.numpy()
    else:
        result = array = np.empty(spec["nbytes"], dtype=np.uint8)
    with path.open("rb") as file:
        file.seek(offset)
        if file.readinto(array) != spec["nbytes"]:
            raise ValueError(f"Incomplete prepared tensor: {path} at {offset}")
    return result


def _fixture_snapshot(args, index, tensor, version, pinned=False):
    spec = index[tensor["name"]]
    return (
        _read_prepared_tensor(args, spec, version - 1, pinned),
        _read_prepared_tensor(args, spec, version, pinned),
    )


def _publication_accounting(publication):
    path = Path(publication["manifest_path"])
    manifest = json.loads(path.read_text())
    payload_bytes = sum(item["nbytes"] for item in manifest["files"])
    inner_bytes = sum(frame["encoded_bytes"] for tensor in manifest["tensors"] for frame in tensor["frames"])
    outer_bytes = sum(tensor.get("outer", {}).get("encoded_bytes", 0) for tensor in manifest["tensors"])
    raw_bytes = sum(tensor.get("raw", {}).get("encoded_bytes", 0) for tensor in manifest["tensors"])
    sizes = {
        "raw_tensor_count": sum(tensor["encoding"] == "raw_bytes" for tensor in manifest["tensors"]),
        "raw_target_bytes": raw_bytes,
        "matrix_payload_bytes": outer_bytes,
        "inner_arena_bytes": sum(tensor.get("outer", {}).get("decoded_bytes", 0) for tensor in manifest["tensors"]),
        "outer_zstd_frames": sum(len(tensor.get("outer", {}).get("frames", [])) for tensor in manifest["tensors"]),
        "encoded_frame_bytes": inner_bytes,
        "payload_file_bytes": payload_bytes,
        "alignment_bytes": payload_bytes - raw_bytes - outer_bytes,
        "manifest_bytes": path.stat().st_size,
        "publication_bytes": payload_bytes + path.stat().st_size,
    }
    return sizes


def _group_key(name):
    if match := re.search(r"(?:^|\.)layers\.(\d+)\.", name):
        return 0, int(match[1])
    if "embed_tokens." in name:
        return 1, 0
    if "lm_head." in name:
        return 2, 0
    return 3, 0


def _partition_plan(plan, owner_count):
    groups = {}
    for tensor in plan:
        groups.setdefault(_group_key(tensor["name"]), []).append(tensor)
    ordered = sorted(groups)
    count, remainder = divmod(len(ordered), owner_count)
    owners, start = [], 0
    for owner in range(owner_count):
        stop = start + count + (owner < remainder)
        keys = ordered[start:stop]
        tensors = [tensor for key in keys for tensor in sorted(groups[key], key=lambda item: item["name"])]
        owners.append(
            {
                "owner": owner,
                "gpu": owner,
                "groups": [
                    f"layer:{key[1]}" if key[0] == 0 else ("embedding", "lm_head", "other")[key[0] - 1] for key in keys
                ],
                "tensors": tensors,
                "canonical_bytes": sum(math.prod(t["shape"]) * DTYPE_BYTES[t["dtype"]] for t in tensors),
            }
        )
        start = stop
    return owners


def _encode_version(args, owner, index, encoder, metadata, version):
    tensors = owner["tensors"]
    raw_plan = [tensor for tensor in tensors if tensor["encoding"] == "raw_bytes"]
    matrix_plan = [tensor for tensor in tensors if tensor["encoding"] == "xor_bytes"]
    writer = PublicationWriter(
        args.output / metadata["codec"] / f"v{version}",
        **metadata,
        owner=owner["owner"],
        base_version=version - 1,
        target_version=version,
        publication_id=f"{metadata['stream_id']}:{version}",
    )
    started, changed, input_read_s = time.monotonic(), 0, 0.0
    pending, batch, batch_bytes = [], [], 0
    batch_wall_s, publication_write_s = 0.0, 0.0
    try:
        for tensor in raw_plan:
            reading = time.monotonic()
            before, after = _fixture_snapshot(args, index, tensor, version)
            input_read_s += time.monotonic() - reading
            writing = time.monotonic()
            entry = writer.add_raw_tensor(
                tensor["name"], before, after, dtype=tensor["dtype"], shape=tensor["shape"], views=tensor["views"]
            )
            publication_write_s += time.monotonic() - writing
            changed += entry["changed_bytes"]
        for tensor in matrix_plan:
            reading = time.monotonic()
            before, after = _fixture_snapshot(args, index, tensor, version, pinned=True)
            input_read_s += time.monotonic() - reading
            batch.append((before, after))
            batch_bytes += after.numel()
            # A batching target, not an overall memory cap: a larger tensor
            # stays whole; compact inner payloads remain until owner finalization.
            if batch_bytes >= 512 * 1024**2:
                pending.extend(encoder.encode_device(batch))
                batch, batch_bytes = [], 0
        if batch:
            pending.extend(encoder.encode_device(batch))
        batch = []
        for tensor, (frames, payload, outer, count, metrics) in zip(
            matrix_plan, encoder.finish_device(pending), strict=True
        ):
            batch_wall_s += metrics["encode_wall_s"]
            writing = time.monotonic()
            writer.add_encoded_tensor(
                tensor["name"],
                frames,
                payload,
                outer,
                changed_bytes=count,
                dtype=tensor["dtype"],
                shape=tensor["shape"],
                views=tensor["views"],
            )
            publication_write_s += time.monotonic() - writing
            changed += count
        writing = time.monotonic()
        shard = writer.finish_shard()
        publication_write_s += time.monotonic() - writing
    finally:
        writer.close()
    finalization = encoder.finalization_metrics
    metrics = {key: owner[key] for key in ("owner", "gpu", "groups", "canonical_bytes")}
    metrics.update(
        pid=os.getpid(),
        batch_wall_s=batch_wall_s,
        finalization=finalization,
        publication_write_s=publication_write_s,
        compression_and_publication_s=batch_wall_s + finalization["finalize_wall_s"] + publication_write_s,
        matrix_hash_write_s=writer.payload_metrics["matrix_hash_write_s"],
        payload_checksum_format=writer.metadata["payload_checksum_format"],
        input_read_s=input_read_s,
        owner_elapsed_s=time.monotonic() - started,
    )
    return {"version": version, "changed_bytes": changed, "shard": shard, "metrics": metrics}


def _encode_owner(args, owner, index, metadata):
    # Spawned workers own CUDA contexts; the parent seals their publications.
    import torch

    from miles.utils.gpu_delta.encoder import GpuBatchEncoder

    results = []
    try:
        torch.cuda.set_device(owner["gpu"])
        encoder = GpuBatchEncoder(
            torch.device("cuda", owner["gpu"]), codec=metadata["codec"], frame_bytes=args.frame_bytes
        )
        for version in range(1, args.versions + 1):
            result = _encode_version(args, owner, index, encoder, metadata, version)
            _save(args.output / f"owner-{owner['owner']:05d}-v{version}.json", result)
            results.append(result)
        return results
    except BaseException as error:
        _save(args.output / f"owner-{owner['owner']:05d}-failed.json", {"error": repr(error)})
        raise


def _fixture(args):
    import torch

    codec = configured_codec()
    inventory = json.loads(args.inventory.read_text())
    plan, digest = merge_plans(inventory["descriptions"])
    index = _read(args.prepared_inputs / "data" / "index.json")
    visible = torch.cuda.device_count()
    owner_count = visible if args.sender_gpus is None else args.sender_gpus
    if not 0 < owner_count <= visible:
        raise ValueError("sender GPUs must be positive and available in CUDA_VISIBLE_DEVICES")
    owners = [owner for owner in _partition_plan(plan, owner_count) if owner["groups"]]
    assignments = [owner | {"tensors": [t["name"] for t in owner["tensors"]]} for owner in owners]
    _save(args.output / "owner-plan.json", assignments)
    metadata = {
        "stream_id": sha256(
            canonical_json({"seed": args.seed, "plan": digest, "rate": args.mutation_rate, "arm": args.arm_name})
        ),
        "plan_digest": digest,
        "codec": codec,
        "frame_bytes": args.frame_bytes,
    }
    if len(owners) == 1:
        results = [_encode_owner(args, owners[0], index, metadata)]
    else:
        with ProcessPoolExecutor(
            max_workers=len(owners), mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
        ) as pool:
            futures = [pool.submit(_encode_owner, args, owner, index, metadata) for owner in owners]
            results = [future.result() for future in futures]
    _seal_fixture(args, plan, metadata, assignments, results)


def _seal_fixture(args, plan, metadata, assignments, results):
    codec = metadata["codec"]
    denominator = sum(math.prod(t["shape"]) * DTYPE_BYTES[t["dtype"]] for t in plan)
    report = {
        "plan_digest": metadata["plan_digest"],
        "rate": args.mutation_rate,
        "requested_zstd_ratio": args.ratio,
        "seed": args.seed,
        "stream_id": metadata["stream_id"],
        "target_checkpoint": str(args.target.resolve()),
        "input_cache_id": args.input_cache_id,
        "rounds": [],
        "calibration": "CPU Zstd level-1 sample-frame estimate; selected-codec size is measured, not forced to this ratio.",
        "canonical_denominator": "Canonical tensors in the receiver plan; excludes frozen draft and other checkpoint entries.",
        "changed_bytes_definition": "Unequal storage bytes, not changed bits or compressed size.",
        "codec": codec,
        "arm": args.arm_name,
        "sender_gpus": len(results),
        "sender_owners": assignments,
        "owner_plan": "owner-plan.json",
        "inner_codec_origin": "Production GpuBatchEncoder on each owner's GPU: pinned snapshots, GPU XOR/inner compression, optional owner-wide GPU Zstd. Contiguous whole-layer HF ownership, no Megatron export.",
        "fixture_wall_scope": "Per-version maximum of owner-local prepared-input read/encode/publication spans. Owners are not synchronized between versions. Parent sealing is separate.",
        "compression_wall_scope": "Sequential owner encode_device, finish_device and publication calls, including optional payload SHA, H2D/XOR/final D2H and shard flush/fsync. Excludes shared input preparation, pinned input reads and parent manifest sealing.",
    }
    for version in range(1, args.versions + 1):
        records = [owner[version - 1] for owner in results]
        started = time.monotonic()
        publication = seal_publication(args.output / codec / f"v{version}", [record["shard"] for record in records])
        seal_s = time.monotonic() - started
        sizes = _publication_accounting(publication)
        changed = sum(record["changed_bytes"] for record in records)
        row = {
            "version": version,
            "publications": {codec: publication},
            "canonical_bytes": denominator,
            "changed_bytes": changed,
            "changed_byte_fraction": changed / denominator,
            "accounting": {codec: sizes},
            "ratios": {codec: sizes["publication_bytes"] / denominator},
            "encoded_frame_ratios": {codec: sizes["encoded_frame_bytes"] / denominator},
            "owner_elapsed_s": max(record["metrics"]["owner_elapsed_s"] for record in records),
            "seal_s": seal_s,
            "compression": {"owners": [record["metrics"] for record in records]},
        }
        report["rounds"].append(row)
        _save(args.output / "fixture.json", report)
        print(json.dumps({key: value for key, value in row.items() if key != "publications"}), flush=True)


async def _request(client, endpoint, payload=None, timeout=1200):
    # Reuse the engine-scoped HTTP client and TLS context across updates.
    http = client.http
    response = await (
        http.get(client.server_url + "/" + endpoint, timeout=timeout)
        if payload is None
        else http.post(client.server_url + "/" + endpoint, json=payload, timeout=timeout)
    )
    response.raise_for_status()
    return response.json() if response.content else None


class _BenchmarkClient:
    """Direct HTTP facade for the standalone producer/receiver benchmark."""

    def __init__(self, server_url):
        self.server_url = server_url
        self.http = None

    async def get_gpu_delta_info(self, engine_id):
        return await _request(self, "get_gpu_delta_info", {"engine_id": engine_id})

    async def update_weights_from_gpu_delta(self, **payload):
        # Preparation has its own server deadline; a client read deadline must
        # not turn a valid slow preparation into an uncertain update.
        return await _request(self, "update_weights_from_gpu_delta", payload, timeout=httpx.Timeout(1200.0, read=None))

    async def clear_gpu_delta_state(self):
        return await _request(self, "clear_gpu_delta_state", {})


async def _ready(client, process, timeout):
    async def poll():
        while True:
            if process.poll() is not None:
                raise RuntimeError("Engine exited during startup; inspect retained server log")
            try:
                response = await client.http.get(client.server_url + "/health", timeout=2)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _freeze_gc(client):
    response = await client.http.post(client.server_url + "/freeze_gc", timeout=60)
    response.raise_for_status()


def _group_alive(pgid):
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
            return True
    return False


def _stop_engine(process):
    # Drain the owned group even if its launcher exited before its workers.
    for sig, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()  # Reap the launcher; do not mistake its zombie for live work.
            if not _group_alive(process.pid):
                process.wait()
                return
            time.sleep(0.1)
    raise RuntimeError(f"Benchmark engine process group {process.pid} did not exit")


@asynccontextmanager
async def _engines(args, model):
    processes, logs, clients = [], [], []
    commands = []
    try:
        specs = _engine_specs(args.ports)
        for engine, spec in enumerate(specs):
            port = spec["port"]
            with socket.socket() as sock:
                # Match the server's reuse policy when a previous arm leaves TIME_WAIT sockets.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
            config = SERVER_ARGS | {
                "tp_size": spec["parallel_size"],
                "dp_size": spec["parallel_size"],
                "ep_size": spec["parallel_size"],
                "model_path": str(model),
                "speculative_draft_model_path": str(args.model),
                "host": "127.0.0.1",
                "port": port,
                "random_seed": 1238,
            }
            config_path = args.output / f"engine-{engine}.json"
            _save(config_path, config)
            code = "import json,sys; from sglang.srt.server_args import ServerArgs; from sglang.srt.entrypoints.http_server import launch_server; launch_server(ServerArgs(**json.load(open(sys.argv[1]))))"
            command = [sys.executable, "-c", code, str(config_path)]
            env = os.environ | SERVER_ENV | {"CUDA_VISIBLE_DEVICES": spec["gpu_ids"]}
            # Miles selects payload hashing and sends it in each update request.
            env.pop("GPU_DELTA_SKIP_PAYLOAD_HASH", None)
            log = (args.output / f"engine-{engine}.log").open("x")
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            processes.append(process)
            logs.append(log)
            client = _BenchmarkClient(f"http://127.0.0.1:{port}")
            clients.append(client)
            client.http = httpx.AsyncClient(trust_env=False, timeout=1200)
            commands.append(spec | {"command": command, "pid": process.pid})
        _save(
            args.output / "launch.json",
            {
                "engines": commands,
                "receiver_validate_payload_hash": not configured_skip_payload_hash(),
                "feature_env": {
                    key: os.environ.get(key)
                    for key in (
                        "GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS",
                        "GPU_DELTA_LAYERS_PER_BATCH",
                        "GPU_DELTA_CPU_WORKERS",
                        "GPU_DELTA_HOST_CACHE_DIR",
                        "SGLANG_ENABLE_NVTX_OPERATIONS",
                    )
                },
            },
        )
        started = time.monotonic()
        await asyncio.gather(*[_ready(c, p, args.startup_timeout) for c, p in zip(clients, processes, strict=True)])
        _save(
            args.output / "startup.json",
            {
                "engine_startup_s": time.monotonic() - started,
                "servers": await asyncio.gather(*[_request(c, "get_server_info") for c in clients]),
            },
        )
        await asyncio.gather(*[_freeze_gc(client) for client in clients])
        yield clients
    finally:
        # Close every transport even after partial startup or a failed request.
        # Only the benchmark's own fresh process groups; the devbox is retained.
        try:
            results = await asyncio.gather(
                *[client.http.aclose() for client in clients if client.http is not None],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        finally:
            try:
                await asyncio.gather(*[asyncio.to_thread(_stop_engine, process) for process in processes])
            finally:
                for log in logs:
                    log.close()


async def _generation(clients):
    records = []
    engine_ids = [f"engine-{i:05d}" for i in range(len(clients))]
    for dp_rank in range(8 // len(clients)):
        payload = {
            "text": "Explain why water freezes at low temperatures in one sentence.",
            "routed_dp_rank": dp_rank,
            "return_logprob": True,
            "logprob_start_len": 0,
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        }
        values = await asyncio.gather(*[_request(client, "generate", payload) for client in clients])
        records.append({"dp_rank": dp_rank, "engine_ids": engine_ids, "engines": values})
    return records


def _profile_update(enabled):
    if not enabled:
        return nullcontext()
    # Only an explicit profiling run imports the NVTX API. The one coordinator
    # gate encloses both engines; rank-local ranges must not control capture.
    from torch.cuda import nvtx

    return nvtx.range("gpu_delta_benchmark_capture")


async def _run(args, phase):
    codec = configured_codec()
    fixture = json.loads((args.fixture / "fixture.json").read_text()) if args.fixture else None
    model = Path(fixture["target_checkpoint"]) if phase == "oracle" else args.model
    async with _engines(args, model) as clients:
        if phase == "oracle":
            _save(args.output / "target-generation.json", await _generation(clients))
            return
        descriptions = await asyncio.gather(
            *[c.get_gpu_delta_info(engine_id=f"engine-{i:05d}") for i, c in enumerate(clients)]
        )
        cohort = negotiate_cohort(descriptions)
        digest = cohort.plan_digest
        _validate_cohort(cohort, len(clients))
        _save(args.output / "gpu-processes.json", _capture_gpu_processes(cohort, args.ports))
        _save(args.output / "inventory.json", {"descriptions": descriptions, "plan_digest": digest})
        if phase == "inventory":
            return
        if digest != fixture["plan_digest"]:
            raise ValueError("Fixture plan no longer matches current loaded models")
        _save(args.output / "base-generation.json", await _generation(clients))
        for version in fixture["rounds"]:
            profiled = version["version"] == args.profile_version
            started = time.monotonic()
            try:
                with _profile_update(profiled):
                    if profiled:
                        from torch.cuda import nvtx

                        # Allow the profiler to start collecting child processes.
                        # The trace still needs review for complete rank coverage.
                        with nvtx.range("gpu_delta.capture_arming"):
                            await asyncio.sleep(0.2)
                        started = time.monotonic()
                    receipt = await activate_publication(clients, cohort, version["publications"][codec])
                result = {
                    "version": version["version"],
                    "coordinator_s": time.monotonic() - started,
                    "codec": codec,
                    "measurement_phase": "first-use-allocation" if version["version"] == 1 else "warm-update",
                    "receipt": receipt,
                }
                if args.profile_version is not None:
                    result["profile"] = {
                        "tool": "external nsys launcher",
                        "capture_version": args.profile_version,
                        "capture_range_emitted": profiled,
                        "nvtx_capture_range": "gpu_delta_benchmark_capture",
                        "capture_arming_s": 0.2 if profiled else 0.0,
                        "capture_arming_scope": "before coordinator timer; trace coverage requires review",
                        "scope": "receiver preparation through all engine resume replies",
                        "timing_scope": "profiling run; exclude from unprofiled performance comparisons",
                    }
                _save(args.output / f"update-{version['version']}.json", result)
            except Exception as error:
                _save(
                    args.output / f"update-{version['version']}-failed.json",
                    {"error": repr(error), "elapsed_s": time.monotonic() - started},
                )
                raise
            # Untimed functional check; exact storage checks belong to primitive
            # tests. Generation equality alone does not prove every weight byte.
            _save(args.output / f"generation-{version['version']}.json", await _generation(clients))
        cleared = await asyncio.gather(*[client.clear_gpu_delta_state() for client in clients])
        _save(args.output / "clear-state.json", cleared)
        if any(not reply["success"] for reply in cleared):
            raise RuntimeError("Receiver state release failed")


def _read(path):
    return json.loads(path.read_text())


def _seconds(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid timing: {value}")
    return value


def _generations(records, routes, version=None):
    selected = {}
    for record in records:
        rank = record["dp_rank"]
        for engine, response in zip(record["engine_ids"], record["engines"], strict=True):
            route = (engine, rank)
            meta, tokens = response["meta_info"], response["output_ids"]
            if route in selected or meta["dp_rank"] != rank:
                raise ValueError("Duplicate or incorrectly routed generation")
            output = meta["output_token_logprobs"]
            if not tokens or len(tokens) != meta["completion_tokens"] or len(output) != len(tokens):
                raise ValueError("Incomplete generation output/logprobs")
            if any(row[1] != token or not math.isfinite(row[0]) for row, token in zip(output, tokens, strict=True)):
                raise ValueError("Unaligned or nonfinite output logprobs")
            inputs = meta["input_token_logprobs"]
            if len(inputs) != meta["prompt_tokens"]:
                raise ValueError("Incomplete prompt logprobs; request logprob_start_len=0")
            if any(not (i == 0 and row[0] is None) and not math.isfinite(row[0]) for i, row in enumerate(inputs)):
                raise ValueError("Nonfinite input logprobs")
            actual_version = str(meta["weight_version"])
            if version is not None and actual_version != str(version):
                raise ValueError("Generation used the wrong weight version")
            end = 0
            for span in meta["weight_versions"]:
                if (
                    span["start"] != end
                    or not end < span["end"] <= len(tokens)
                    or str(span["version"]) != actual_version
                ):
                    raise ValueError("Mixed or incomplete generation version spans")
                end = span["end"]
            if end != len(tokens) or meta["prompt_tokens"] <= 0:
                raise ValueError("Incomplete generation version/prompt coverage")
            selected[route] = {
                "text": response["text"],
                "output_ids": tokens,
                "prompt_tokens": meta["prompt_tokens"],
                "input_token_logprobs": inputs,
                "output_token_logprobs": output,
            }
    if set(selected) != routes:
        raise ValueError("Missing or unexpected generation routes")
    return selected


def _compare(left, right):
    rows = []
    for engine, rank in sorted(left):
        a, b = left[engine, rank], right[engine, rank]
        equal = {key: a[key] == b[key] for key in a}
        rows.append({"engine_id": engine, "dp_rank": rank, "equal": equal, "exact": all(equal.values())})
    return {"status": "PASS" if all(row["exact"] for row in rows) else "FAILED", "routes": rows}


def _apply_metrics(timings):
    """Optional kernel/setup measurements; old receipts remain unmeasured."""
    fields = {
        "kernel_prepare_s": "host_apply_kernel_prepare_s",
        "kernel_cache_misses": "apply_kernel_cache_misses",
        "kernel_cache_reused": "apply_kernel_cache_reused",
        "setup_s": "paused_setup_host_s",
        "apply_host_s": "paused_apply_host_wall_s",
        "descriptor_h2d_bytes": "apply_descriptor_h2d_bytes",
        "apply_metadata_h2d_bytes": "apply_metadata_h2d_bytes",
    }
    result = {name: _seconds(timings[field]) if field in timings else None for name, field in fields.items()}
    # Three fixed kernel families can each miss once; reuse is a boolean counter.
    for name, allowed in (("kernel_cache_misses", (0, 1, 2, 3)), ("kernel_cache_reused", (0, 1))):
        if result[name] is not None and result[name] not in allowed:
            raise ValueError(f"Invalid per-update kernel counter: {name}")
    return result


def _rank_rows(update, publication, identities):
    receipt = update["receipt"]
    stages = []
    for key, state in (("receipts", "APPLIED"), ("resumed_receipts", "RESUMED")):
        by_rank = {row["identity"]["rank_id"]: row for row in receipt[key]}
        if len(by_rank) != len(receipt[key]) or set(by_rank) != set(identities):
            raise ValueError("Missing or duplicate rank receipts")
        for rank, row in by_rank.items():
            if row["identity"] != identities[rank] or row["state"] != state or not row["result"]["applied"]:
                raise ValueError("Rank identity or apply/resume state differs")
            if row["session_id"] != receipt["session_id"]:
                raise ValueError("Rank session differs")
            for field in ("manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest"):
                if row[field] != publication[field]:
                    raise ValueError(f"Rank publication differs: {field}")
            if row["result"]["target_version"] != publication["target_version"]:
                raise ValueError("Applied target differs")
        stages.append(by_rank)
    rows = []
    for rank, resumed in stages[1].items():
        timing = resumed["scheduler_timing"]
        start, fence, end = (timing[key] for key in ("pause_started_ns", "reader_fence_completed_ns", "resumed_ns"))
        if resumed["generation_paused"] or end is None or not start <= fence <= end:
            raise ValueError("Scheduler pause did not complete")
        pause = _seconds(timing["blocked_s"])
        if not math.isclose(pause, (end - start) / 1e9, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Scheduler pause endpoints differ from reported duration")
        if start != stages[0][rank]["scheduler_timing"]["pause_started_ns"]:
            raise ValueError("Apply and resume pause endpoints differ")
        result, metrics = resumed["result"], resumed["result"]["timings"]
        # Older receipts may contain CUDA-event diagnostics; fresh runs use Nsight.
        cuda = metrics["cuda_event_ms"] if result.get("timing_enabled", False) else None
        rows.append(
            {
                "identity": identities[rank],
                "outer_cpu_s": _seconds(metrics["host_rank_outer_zstd_decode_s"]),
                "plain_copy_s": _seconds(metrics["host_rank_encoded_copy_s"]),
                "de_stream_cuda_s": _seconds(cuda["decode"]) / 1000 if cuda is not None else None,
                "matrix_apply_cuda_s": _seconds(cuda["layout_apply"]) / 1000 if cuda is not None else None,
                "gpu_pipeline_s": (
                    _seconds(cuda["paused_gpu_pipeline"]) / 1000
                    if cuda is not None and "paused_gpu_pipeline" in cuda
                    else None
                ),
                "prepare_s": _seconds(metrics["host_prepare_s"]),
                "pause_s": pause,
                "skip_payload_hash": metrics["host_encoded_cache_skip_payload_hash"],
                "receipt": resumed,
                **_apply_metrics(metrics),
            }
        )
    return sorted(rows, key=lambda row: (row["identity"]["engine_id"], row["identity"]["dp_rank"]))


def _owner_compression(metrics, codec):
    final = metrics["finalization"]
    # Saved fixtures retain their original diagnostics, never new measurements.
    inner_cuda = metrics.get("batch_cuda_s", {})
    outer_cuda = final.get("finalize_cuda_phase_s", {})
    inner = _seconds(inner_cuda["compression_s"]) if "compression_s" in inner_cuda else None
    outer = _seconds(outer_cuda["outer_zstd_s"]) if codec != "lz4" and "outer_zstd_s" in outer_cuda else None
    combined = _seconds(metrics["compression_and_publication_s"])
    if not math.isclose(
        combined,
        metrics["batch_wall_s"] + final["finalize_wall_s"] + metrics["publication_write_s"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("Compression/publication timing differs from its sequential components")
    return {
        "compression_and_publication_s": combined,
        "publication_write_s": _seconds(metrics["publication_write_s"]),
        "inner_cuda_s": inner,
        "outer_cuda_s": outer,
        "batch_wall_s": _seconds(metrics["batch_wall_s"]),
        "finalization_wall_s": _seconds(final["finalize_wall_s"]),
        "outer_wall_s": _seconds(final["outer_zstd_wall_s"]),
        "pack_d2h_wall_s": _seconds(final["pack_d2h_wall_s"]),
        "input_read_s": _seconds(metrics["input_read_s"]),
        "owner_elapsed_s": _seconds(metrics["owner_elapsed_s"]),
        "raw_metrics": metrics,
    }


def _compression(row, codec, assignments):
    metrics = row["compression"]["owners"]
    by_owner = {owner["owner"]: owner for owner in metrics}
    if len(by_owner) != len(metrics) or set(by_owner) != set(assignments):
        raise ValueError("Missing or duplicate compression owners")
    for owner, value in by_owner.items():
        if any(value[key] != assignments[owner][key] for key in ("gpu", "groups", "canonical_bytes")):
            raise ValueError("Compression owner assignment differs")
    owners = [_owner_compression(value, codec) for value in metrics]
    if not owners or sum(owner["raw_metrics"]["canonical_bytes"] for owner in owners) != row["canonical_bytes"]:
        raise ValueError("Compression owners do not cover the canonical bytes")
    return {
        "owner_max": {
            key: _maximum(owners, key)
            for key in (
                "compression_and_publication_s",
                "publication_write_s",
                "inner_cuda_s",
                "outer_cuda_s",
                "batch_wall_s",
                "finalization_wall_s",
                "outer_wall_s",
                "pack_d2h_wall_s",
                "input_read_s",
                "owner_elapsed_s",
            )
        },
        "seal_s": _seconds(row["seal_s"]),
        "owners": owners,
    }


def _maximum(rows, key):
    values = [row[key] for row in rows]
    return None if any(value is None for value in values) else max(values)


def _format(value, scale=1):
    return "unmeasured" if value is None else f"{value * scale:.3f}"


def _report_table(headers, rows):
    return [
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
        *("|" + "|".join(map(str, row)) + "|" for row in rows),
        "",
    ]


def _markdown(summary):
    versions = [(arm, row) for arm in summary["arms"] for row in arm["versions"]]
    lines = [
        "# GPU delta end-to-end benchmark",
        "",
        f"Status: **{summary['status']}**. "
        + (
            "Final target oracle NOT_RUN: no target-equivalence or full correctness claim. Apply/resume receipts and per-update ordinary generations were checked."
            if summary.get("final_target_validation") == "NOT_RUN"
            else "Every arm's final selected text, token IDs and full input/output logprobs are compared with one independently loaded target checkpoint on every route; see `comparison.json`."
        ),
        "",
        "One row per arm/version, including the first update. SHA on/off controls both sender payload "
        "hashing and receiver verification; manifest SHA remains enabled. Versions are cumulative synthetic "
        "targets, not learned updates or statistical repeats. Selected outputs do not prove every weight byte. "
        "Startup and oracle generation are outside the update timings.",
        "",
        f"Prepared inputs: `{summary['inputs']['path']}`; reused: {summary['inputs']['reused']}. Input construction is shared across all arms and excluded from compression measurements; its metadata remains in `inputs.json`.",
        "",
    ]
    if any("profile" in row for _, row in versions):
        lines += [
            "Only the version marked **capture range** emitted the NVTX capture range. All rows in this "
            "profiling run used instrumented processes; their receiver timings are separate from unprofiled performance results. "
            "The selected range includes a 0.2-second capture-arming lead-in before the coordinator timer; "
            "this delay does not prove complete trace coverage.",
            "",
        ]
    lines += _report_table(
        [
            "Codec",
            "Frame KiB",
            "SHA",
            "Version",
            "Sender timing origin",
            "Compression s",
            "Prepare s",
            "Pause ms",
            "Inner GiB",
            "Outer GiB",
            "Payload GiB",
        ],
        (
            [
                arm["codec"],
                row["frame_bytes"] / 1024,
                "off" if arm["skip_payload_hash"] else "on",
                (
                    f"{row['version']} ({'capture range' if row['profile']['capture_range_emitted'] else 'profiling run'})"
                    if "profile" in row
                    else row["version"]
                ),
                row["compression"].get("measurement_scope", "current_run"),
                _format(row["compression"]["owner_max"]["compression_and_publication_s"]),
                _format(row["receiver_max"]["prepare_s"]),
                _format(row["receiver_max"]["pause_s"], scale=1000),
                _format(row["accounting"]["encoded_frame_bytes"], scale=1 / 2**30),
                (
                    "absent"
                    if arm["codec"] == "lz4"
                    else _format(row["accounting"]["matrix_payload_bytes"], scale=1 / 2**30)
                ),
                _format(row["accounting"]["payload_file_bytes"], scale=1 / 2**30),
            ]
            for arm, row in versions
        ),
    )
    lines += [
        "In saved-delta apply mode, every sender/read/encode/seal/compression timing is historical from "
        "the referenced fixture, never current-run encoder performance. Validation I/O is recorded separately "
        "in fixture-verification.json; no encoding time is invented or included in current totals. "
        "New receiver apply/resume/generation measurements remain current. The final oracle is opt-in.",
        "",
        "Compression is the maximum owner compression/publication wall time, including optional payload SHA, "
        "H2D/XOR/final D2H, writes and shard flush/fsync. It excludes prepared-input reads/pinning, shared input "
        "construction and parent manifest sealing. Prepare is the maximum receiver background-preparation "
        "wall time. Pause is the maximum actual scheduler pause from completed pause/resume timestamps. "
        "These independent owner/rank maxima can come from different processes and must not be added.",
        "",
        "Sizes sum across all owners; GiB = 2^30 bytes. Inner is the compressed frame bytes before outer Zstd. "
        "Outer is the stored Zstd matrix payload; it is absent for plain LZ4. Payload includes all stored "
        "matrix/raw bytes and alignment, excluding the manifest. The packed inner arena, exact byte counts, "
        "manifest/publication sizes, frame sizes and canonical/changed bytes remain in `summary.json`.",
        "",
        "`summary.json` retains every owner/rank receipt at full precision, including input read/pin and "
        "owner elapsed time, publication writes, finalization, pack/D2H, parent sealing, "
        "CPU outer decode/plain copy and coordinator time. Components overlap. CUDA intervals from "
        "older fixtures/receipts remain historical diagnostics, including wrapper/launch gaps; fresh runs "
        "leave them unmeasured and use Nsight for GPU profiling. The table rounds values to three decimal places.",
        "",
    ]
    lines += _report_table(
        [
            "Codec",
            "Version",
            "Kernel prepare ms",
            "Kernel cache misses/rank",
            "Paused setup ms",
            "Apply wall ms",
            "Layout CUDA ms",
            "GPU pipeline ms",
            "Descriptor KiB",
            "Apply metadata KiB",
        ],
        (
            [
                arm["codec"],
                row["version"],
                _format(row["receiver_max"]["kernel_prepare_s"], scale=1000),
                _format(row["receiver_max"]["kernel_cache_misses"]),
                _format(row["receiver_max"]["setup_s"], scale=1000),
                _format(row["receiver_max"]["apply_host_s"], scale=1000),
                _format(row["receiver_max"]["matrix_apply_cuda_s"], scale=1000),
                _format(row["receiver_max"]["gpu_pipeline_s"], scale=1000),
                _format(row["receiver_max"]["descriptor_h2d_bytes"], scale=1 / 1024),
                _format(row["receiver_max"]["apply_metadata_h2d_bytes"], scale=1 / 1024),
            ]
            for arm, row in versions
        ),
    )
    lines += [
        "Kernel preparation occurs before the serving pause and includes the cached kernel lookup or first "
        "compilation/loading. Cache misses count per-process module warmup requests per update; a request may compile or load a disk-cached binary. Missing counters in older "
        "receipts remain unmeasured. Descriptor bytes are included in total apply metadata bytes. Each cell is "
        "an independent rank maximum; nested host spans and overlapping CUDA events must not be added.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _arm_report(output, arm, oracle_records):
    fixture_directory, reuse = _fixture_reuse.resolve(output)
    fixture = _read(fixture_directory / "fixture.json")
    codec = arm["codec"]
    checksum = "none" if arm["skip_payload_hash"] else "sha256"
    if fixture["codec"] != codec or fixture["arm"] != arm["name"]:
        raise ValueError("Fixture codec/arm differs")
    sender_owners = fixture["sender_owners"]
    assignments = {owner["owner"]: owner for owner in sender_owners}
    if len(assignments) != len(sender_owners) or len(assignments) != fixture["sender_gpus"]:
        raise ValueError("Sender owner plan differs from the fixture")
    inventory = _read(output / "receiver" / "inventory.json")
    if inventory["plan_digest"] != fixture["plan_digest"]:
        raise ValueError("Receiver and fixture plans differ")
    original = [rank["identity"] for engine in inventory["descriptions"] for rank in engine["participants"]]
    identities = {identity["rank_id"]: identity for identity in original}
    routes = {(identity["engine_id"], identity["dp_rank"]) for identity in original}
    if not identities or len(identities) != len(original) or len(routes) != len(original):
        raise ValueError("Missing or duplicate receiver identities/routes")
    versions = []
    for expected, row in enumerate(fixture["rounds"], 1):
        version = row["version"]
        publication = row["publications"][codec]
        if (
            version != expected
            or publication["base_version"] != version - 1
            or publication["target_version"] != version
        ):
            raise ValueError("Fixture versions are not consecutive")
        if publication["payload_checksum_format"] != checksum:
            raise ValueError("Sender publication hash policy differs from arm")
        update = _read(output / "receiver" / f"update-{version}.json")
        if update["version"] != version or update["codec"] != codec:
            raise ValueError("Update version/codec differs")
        ranks = _rank_rows(update, publication, identities)
        if any(rank["skip_payload_hash"] != arm["skip_payload_hash"] for rank in ranks):
            raise ValueError("Receiver hash policy differs from arm")
        compression = _compression(row, codec, assignments)
        compression["measurement_scope"] = "historical_saved_fixture" if reuse else "current_run"
        if any(owner["raw_metrics"]["payload_checksum_format"] != checksum for owner in compression["owners"]):
            raise ValueError("Sender owner hash policy differs from arm")
        selected = _generations(_read(output / "receiver" / f"generation-{version}.json"), routes, version)
        versions.append(
            {
                "version": version,
                "frame_bytes": publication["frame_bytes"],
                "sender_payload_checksum_format": checksum,
                "receiver_skip_payload_hash": arm["skip_payload_hash"],
                "canonical_bytes": row["canonical_bytes"],
                "changed_bytes": row["changed_bytes"],
                "accounting": row["accounting"][codec],
                "compression": compression,
                "coordinator_s": _seconds(update["coordinator_s"]),
                "receiver_max": {
                    key: _maximum(ranks, key)
                    for key in (
                        "outer_cpu_s",
                        "plain_copy_s",
                        "de_stream_cuda_s",
                        "matrix_apply_cuda_s",
                        "gpu_pipeline_s",
                        "prepare_s",
                        "pause_s",
                        "kernel_prepare_s",
                        "kernel_cache_misses",
                        "kernel_cache_reused",
                        "setup_s",
                        "apply_host_s",
                        "descriptor_h2d_bytes",
                        "apply_metadata_h2d_bytes",
                    )
                },
                "ranks": ranks,
            }
        )
        if "profile" in update:
            versions[-1]["profile"] = update["profile"]
    if not versions:
        raise ValueError("No fixture updates")
    comparison = (
        _compare(selected, _generations(oracle_records, routes))
        if oracle_records is not None
        else {"status": "NOT_RUN", "reason": "Final target oracle disabled; use --validate-final-target"}
    )
    return arm | {
        "status": "NOT_VALIDATED" if comparison["status"] == "NOT_RUN" else comparison["status"],
        "fixture_origin": reuse or {"mode": "fresh_encoding", "path": str(fixture_directory)},
        "update_checks": "PASS",
        "plan_digest": fixture["plan_digest"],
        "seed": fixture["seed"],
        "rate": fixture["rate"],
        "stream_id": fixture["stream_id"],
        "target_checkpoint": fixture["target_checkpoint"],
        "input_cache_id": fixture["input_cache_id"],
        "sender_count": fixture["sender_gpus"],
        "sender_owners": sender_owners,
        "comparison": comparison,
        "versions": versions,
    }


def _report_geometry(report):
    return [
        (row["version"], row["frame_bytes"], row["canonical_bytes"], row["changed_bytes"])
        for row in report["versions"]
    ]


def _write_report(output: Path, arms, validate_final_target=True):
    oracle = _read(output / "oracle" / "target-generation.json") if validate_final_target else None
    reports = [_arm_report(output / "arms" / arm["name"], arm, oracle) for arm in arms]
    reference = reports[0]
    for arm in reports[1:]:
        if any(
            arm[key] != reference[key]
            for key in ("plan_digest", "seed", "rate", "target_checkpoint", "input_cache_id", "sender_owners")
        ):
            raise ValueError("Arms do not share the same target and sender ownership")
        if _report_geometry(arm) != _report_geometry(reference):
            raise ValueError("Arms do not share the same update geometry")
    status = (
        ("PASS" if all(arm["status"] == "PASS" for arm in reports) else "FAILED")
        if validate_final_target
        else "NOT_VALIDATED"
    )
    inputs = _read(output / "inputs.json")
    if reference["input_cache_id"] != inputs["id"]:
        raise ValueError("Arm inputs differ from the prepared-input reference")
    summary = {
        "status": status,
        "inputs": inputs,
        "arms": reports,
        "final_target_validation": status if validate_final_target else "NOT_RUN",
    }
    _save(
        output / "comparison.json",
        {
            "status": status,
            "arms": [{"name": arm["name"]} | arm["comparison"] for arm in reports],
        },
    )
    _save(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_markdown(summary))
    if status == "FAILED":
        raise ValueError("Final selected generation differs from target checkpoint; see comparison.json")
    return summary


def _phase(phase, args, directory, settings, arm):
    os.setsid()  # The parent forwards Ctrl-C once while this process drains its workers.
    # Process exit releases each phase's CUDA context before the next phase starts.
    os.environ.update(settings)
    stage_output = args.output / directory
    stage_output.parent.mkdir(parents=True, exist_ok=True)
    with stage_output.with_suffix(".log").open("x") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        stage = argparse.Namespace(
            **(
                vars(args)
                | {
                    "output": stage_output,
                    "inventory": args.prepared_inputs / "inventory" / "inventory.json",
                    "fixture": (
                        _fixture_reuse.resolve(args.output / "arms" / arm["name"])[0]
                        if phase in {"run", "oracle"}
                        else None
                    ),
                    "target": args.prepared_inputs / "target-checkpoint",
                    "arm_name": arm["name"],
                }
            )
        )
        stage.output.mkdir()
        if phase == "prepare":
            _prepare_inputs(stage)
        elif phase == "fixture":
            _fixture(stage)
        else:
            asyncio.run(_run(stage, phase))


def _run_phase(phase, args, directory, arm, records):
    settings = {
        "GPU_DELTA_CODEC": arm["codec"],
        "GPU_DELTA_SKIP_PAYLOAD_HASH": "1" if arm["skip_payload_hash"] else "0",
    }
    if phase == "run" and args.profile_version is not None:
        settings["SGLANG_ENABLE_NVTX_OPERATIONS"] = "1"
    process = multiprocessing.get_context("spawn").Process(
        target=_phase, args=(phase, args, directory, settings, arm), name=phase
    )
    record = {"phase": phase, "arm": arm["name"], "environment": settings, "status": "RUNNING"}
    records.append(record)
    _save(args.output / "workflow.json", {"arms": args.arms, "phases": records})
    print(f"{phase}: {args.output / directory}", flush=True)
    started = time.monotonic()
    process.start()
    record["pid"] = process.pid
    _save(args.output / "workflow.json", {"arms": args.arms, "phases": records})
    try:
        process.join()
    except KeyboardInterrupt:
        if process.is_alive():
            os.kill(process.pid, signal.SIGINT)
        process.join()
        raise
    record.update(
        exit_code=process.exitcode,
        wall_s=time.monotonic() - started,
        status="PASS" if process.exitcode == 0 else "FAILED",
    )
    _save(args.output / "workflow.json", {"arms": args.arms, "phases": records})
    if process.exitcode != 0:
        raise RuntimeError(f"{phase} failed; see {(args.output / directory).with_suffix('.log')}")


def _end_to_end(args):
    records = []
    if args.sender_gpus is None:
        import torch

        args.sender_gpus = torch.cuda.device_count()
    reused = args.prepared_inputs.exists()
    if args.reuse_fixtures and not reused:
        raise ValueError("Saved-delta apply requires the preserved prepared-input cache; encoding is disabled")
    if not reused:
        args.prepared_inputs.mkdir(parents=True)
        _run_phase("inventory", args, args.prepared_inputs / "inventory", args.arms[0], records)
        _run_phase("prepare", args, args.prepared_inputs / "data", args.arms[0], records)
    prepared = _load_prepared_inputs(args)
    args.mutation_rate = prepared["rate"]
    args.input_cache_id = sha256((args.prepared_inputs / "READY.json").read_bytes())
    _save(
        args.output / "inputs.json",
        {"path": str(args.prepared_inputs), "id": args.input_cache_id, "reused": reused, "preparation": prepared},
    )
    _save(args.output / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    fixture_refs = {}
    if not args.reuse_fixtures:
        _fixture_reuse.record_producer(args, prepared)
    if args.reuse_fixtures:
        from sglang.srt.weight_sync.gpu_delta.host import validate_codec

        started = time.monotonic()
        plan, _ = merge_plans(_read(args.prepared_inputs / "inventory" / "inventory.json")["descriptions"])
        owners = [o for o in _partition_plan(plan, args.sender_gpus) if o["groups"]]
        assignments = [o | {"tensors": [t["name"] for t in o["tensors"]]} for o in owners]
        fixture_refs = _fixture_reuse.admit(args, prepared, plan, assignments, validate_codec, _publication_accounting)
        _save(
            args.output / "fixture-verification.json",
            {
                "status": "VERIFIED",
                "wall_s": time.monotonic() - started,
                "scope": "Current saved-publication metadata and payload SHA verification; no encoding",
            },
        )
    for arm in args.arms:
        directory = Path("arms") / arm["name"]
        if fixture_refs:
            arm_output = args.output / directory
            arm_output.mkdir(parents=True)
            with (arm_output / "fixture-reuse.json").open("x") as file:
                json.dump(fixture_refs[arm["name"]], file, indent=2, allow_nan=False)
                file.write("\n")
            records.append(
                {
                    "phase": "fixture",
                    "arm": arm["name"],
                    "status": "REUSED",
                    "encoding": "SKIPPED_ENCODING",
                    "wall_s": None,
                    "reference": str(directory / "fixture-reuse.json"),
                }
            )
            _save(args.output / "workflow.json", {"arms": args.arms, "phases": records})
        else:
            _run_phase("fixture", args, directory / "fixture", arm, records)
        _run_phase("run", args, directory / "receiver", arm, records)
    if args.validate_final_target:
        _run_phase("oracle", args, Path("oracle"), args.arms[0], records)
    else:
        records.append({"phase": "oracle", "status": "NOT_RUN", "reason": "--validate-final-target not selected"})
        _save(args.output / "workflow.json", {"arms": args.arms, "phases": records})
    _write_report(args.output, args.arms, validate_final_target=args.validate_final_target)
    print((args.output / "REPORT.md").read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Existing NVFP4 HF serving checkpoint")
    parser.add_argument("--output", type=Path, help="New output directory; generated automatically if omitted")
    parser.add_argument(
        "--prepared-inputs", type=Path, help="Create or reuse a completed input cache; defaults to OUTPUT/inputs"
    )
    parser.add_argument(
        "--reuse-fixtures", type=Path, help="Saved-delta apply mode: existing encoder run root; skip encoding"
    )
    parser.add_argument(
        "--reuse-producer-spec",
        type=Path,
        help="Reviewed producer-spec.json or provenance adapter (not the receiver run)",
    )
    parser.add_argument("--reuse-producer-spec-sha256", help="Reviewed SHA256 of the original producer SPEC")
    parser.add_argument(
        "--validate-final-target",
        action="store_true",
        help="Load the full target checkpoint and compare selected outputs on all routes (expensive)",
    )
    parser.add_argument("--codecs", choices=CODECS, nargs="+", default=list(CODECS), help="Sender/receiver codecs")
    parser.add_argument("--frame-bytes", type=int, default=FRAME_BYTES, help="Inner frame bytes; at most 4 MiB")
    parser.add_argument("--sender-gpus", type=int, help="Compression GPUs; defaults to all visible GPUs")
    parser.add_argument(
        "--hashing",
        choices=("on", "off"),
        nargs="+",
        default=["on", "off"],
        help="Payload SHA policy on both sender and receiver",
    )
    parser.add_argument("--versions", type=int, default=3)
    parser.add_argument(
        "--profile-version",
        type=int,
        help="Emit one gpu_delta_benchmark_capture NVTX range around warm version N; use an external nsys launcher",
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument(
        "--ratio", type=float, default=0.002, help="CPU Zstd calibration target for synthetic mutation rate"
    )
    parser.add_argument(
        "--ports", type=int, nargs="+", default=[31000, 31100], help="Two ports: two EP4 engines; one port: EP8"
    )
    parser.add_argument("--startup-timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.reuse_fixtures:
        if not all(
            (args.prepared_inputs, args.reuse_producer_spec, args.reuse_producer_spec_sha256, args.sender_gpus)
        ):
            parser.error("--reuse-fixtures requires --prepared-inputs, --sender-gpus and both producer SPEC options")
        if args.hashing != ["on"]:
            parser.error("Saved-delta apply currently requires --hashing on (authenticated saved payloads)")
        for path in (args.reuse_fixtures, args.reuse_producer_spec, args.prepared_inputs):
            _fixture_reuse.safe_path(path)
    elif args.reuse_producer_spec or args.reuse_producer_spec_sha256:
        parser.error("Producer SPEC options require --reuse-fixtures")
    args.model = args.model.resolve(strict=True)
    args.output = (
        args.output or Path("gpu-delta-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
    ).resolve()
    args.prepared_inputs = (args.prepared_inputs or args.output / "inputs").resolve()
    args.arms = [
        {"name": f"{codec}-hash-{hashing}", "codec": codec, "skip_payload_hash": hashing == "off"}
        for codec in dict.fromkeys(args.codecs)
        for hashing in dict.fromkeys(args.hashing)
    ]
    if args.profile_version is not None:
        if not args.reuse_fixtures or len(args.arms) != 1 or args.validate_final_target:
            parser.error("--profile-version requires saved fixtures, one arm and no final-target oracle")
        if not 2 <= args.profile_version <= args.versions:
            parser.error("--profile-version must select a warm update from 2 through --versions")
    if not 0 < args.frame_bytes <= 4 << 20:
        parser.error("--frame-bytes must be positive and at most 4 MiB")
    if args.sender_gpus is not None and args.sender_gpus < 1:
        parser.error("--sender-gpus must be positive")
    try:
        _engine_specs(args.ports)
    except ValueError as error:
        parser.error(str(error))
    if args.versions < 1 or not 0 < args.ratio < 0.1:
        parser.error("versions must be positive and ratio in (0, 0.1)")
    if args.prepared_inputs == args.output or args.prepared_inputs in args.output.parents:
        parser.error("--output must be outside the prepared-input cache")
    args.output.mkdir(parents=True, exist_ok=False)
    _save(args.output / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    _end_to_end(args)


if __name__ == "__main__":
    main()
