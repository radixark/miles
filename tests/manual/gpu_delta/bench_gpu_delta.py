"""Benchmark GPU-delta compression and receiver updates from one NVFP4 HF base.

Run from the Miles checkout with paired SGLang on PYTHONPATH::

    python tests/manual/gpu_delta/bench_gpu_delta.py --model /models/base

Requires eight Blackwell GPUs, CUDA 13 and nvCOMP >=5.3. The GLM5.2 W4A16
workload uses static bundled MTP and two independent TP4/DP4/EP4 receivers.
Sender processes split whole layers across GPUs; no Megatron checkpoint is needed.
Results include actual compression, preparation, decompression and scheduler-pause
measurements, plus a final selected-output comparison with a disk-loaded target.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

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
    if len(cohort.identities) != 8 or len(cohort.participants) != engine_count:
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


def _alter_fixture_tensor(target, index, tensor, version, seed, rate):
    spec = index[tensor["name"]]
    with (target / spec["shard"]).open("r+b") as file:
        file.seek(spec["offset"])
        before = np.frombuffer(file.read(spec["nbytes"]), dtype=np.uint8)
        if before.size != spec["nbytes"]:
            raise ValueError("Incomplete checkpoint tensor")
        after = _mutate(before, name=tensor["name"], dtype=spec["dtype"], seed=seed, version=version, rate=rate)
        file.seek(spec["offset"])
        file.write(after)
    return before, after


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


def _encode_version(args, owner, index, target, rate, encoder, metadata, version):
    import torch

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
    started, changed = time.monotonic(), 0
    pending, batch, batch_bytes = [], [], 0
    batch_cuda_s, batch_wall_s = {}, 0.0
    try:
        for tensor in raw_plan:
            before, after = _alter_fixture_tensor(target, index, tensor, version, args.seed, rate)
            entry = writer.add_raw_tensor(
                tensor["name"], before, after, dtype=tensor["dtype"], shape=tensor["shape"], views=tensor["views"]
            )
            changed += entry["changed_bytes"]
        for tensor in matrix_plan:
            before, after = _alter_fixture_tensor(target, index, tensor, version, args.seed, rate)
            batch.append((torch.from_numpy(before.copy()).pin_memory(), torch.from_numpy(after).pin_memory()))
            batch_bytes += after.nbytes
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
            for name, seconds in metrics["cuda_phase_s"].items():
                batch_cuda_s[name] = batch_cuda_s.get(name, 0.0) + seconds
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
            changed += count
        shard = writer.finish_shard()
    finally:
        writer.close()
    finalization = encoder.finalization_metrics
    if encoder.timing:
        batch_cuda_s.setdefault("compression_s", 0.0)
        if not finalization["final_payload_d2h_bytes"]:
            finalization = finalization | {"finalize_cuda_phase_s": {"outer_zstd_s": 0.0, "pack_d2h_s": 0.0}}
    metrics = {key: owner[key] for key in ("owner", "gpu", "groups", "canonical_bytes")}
    metrics.update(
        pid=os.getpid(),
        timing_enabled=encoder.timing,
        batch_cuda_s=batch_cuda_s,
        batch_wall_s=batch_wall_s,
        finalization=finalization,
        encode_and_target_write_s=time.monotonic() - started,
    )
    return {"version": version, "changed_bytes": changed, "shard": shard, "metrics": metrics}


def _encode_owner(args, owner, index, target, rate, metadata):
    # Spawned workers own CUDA contexts; the parent only copies and seals files.
    import torch

    from miles.utils.gpu_delta.encoder import GpuBatchEncoder

    results = []
    try:
        torch.cuda.set_device(owner["gpu"])
        encoder = GpuBatchEncoder(
            torch.device("cuda", owner["gpu"]), codec=metadata["codec"], frame_bytes=args.frame_bytes
        )
        for version in range(1, args.versions + 1):
            result = _encode_version(args, owner, index, target, rate, encoder, metadata, version)
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
    plan, _, digest = merge_plans(inventory["descriptions"])
    index = _tensor_index(args.model)
    for tensor in plan:
        actual = index[tensor["name"]]
        if tensor["shape"] != actual["shape"] or tensor["dtype"] != actual["dtype"]:
            raise ValueError(f"Inventory/checkpoint mismatch: {tensor['name']}")
    visible = torch.cuda.device_count()
    owner_count = visible if args.sender_gpus is None else args.sender_gpus
    if not 0 < owner_count <= visible:
        raise ValueError("sender GPUs must be positive and available in CUDA_VISIBLE_DEVICES")
    owners = [owner for owner in _partition_plan(plan, owner_count) if owner["groups"]]
    assignments = [owner | {"tensors": [t["name"] for t in owner["tensors"]]} for owner in owners]
    _save(args.output / "owner-plan.json", assignments)
    target = args.output / "altered-checkpoint"
    target.mkdir()
    # Each owner changes disjoint tensor byte ranges in this independent copy.
    for path in args.model.iterdir():
        if path.is_file() and path.suffix in {".json", ".py", ".model", ".tiktoken", ".safetensors"}:
            shutil.copy2(path, target / path.name)
    rate = _calibrate(plan, args.seed, args.ratio)
    metadata = {
        "stream_id": sha256(canonical_json({"seed": args.seed, "plan": digest, "rate": rate})),
        "plan_digest": digest,
        "codec": codec,
        "frame_bytes": args.frame_bytes,
    }
    if len(owners) == 1:
        results = [_encode_owner(args, owners[0], index, target, rate, metadata)]
    else:
        with ProcessPoolExecutor(
            max_workers=len(owners), mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
        ) as pool:
            futures = [pool.submit(_encode_owner, args, owner, index, target, rate, metadata) for owner in owners]
            results = [future.result() for future in futures]
    _seal_fixture(args, plan, metadata, target, rate, assignments, results)


def _seal_fixture(args, plan, metadata, target, rate, assignments, results):
    codec = metadata["codec"]
    denominator = sum(math.prod(t["shape"]) * DTYPE_BYTES[t["dtype"]] for t in plan)
    report = {
        "plan_digest": metadata["plan_digest"],
        "rate": rate,
        "requested_zstd_ratio": args.ratio,
        "stream_id": metadata["stream_id"],
        "target_checkpoint": str(target.resolve()),
        "rounds": [],
        "calibration": "CPU Zstd level-1 sample-frame estimate; selected-codec size is measured, not forced to this ratio.",
        "canonical_denominator": "Canonical tensors in the receiver plan; excludes frozen draft and other checkpoint entries.",
        "changed_bytes_definition": "Unequal storage bytes, not changed bits or compressed size.",
        "codec": codec,
        "sender_gpus": len(results),
        "sender_owners": assignments,
        "owner_plan": "owner-plan.json",
        "inner_codec_origin": "Production GpuBatchEncoder on each owner's GPU: pinned snapshots, GPU XOR/inner compression, optional owner-wide GPU Zstd. Contiguous whole-layer HF ownership, no Megatron export.",
        "fixture_wall_scope": "Per-version maximum of owner-local encode/target-write spans; owners are not synchronized between versions. Parent sealing is separate.",
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
            "encode_and_target_write_s": max(record["metrics"]["encode_and_target_write_s"] for record in records),
            "seal_s": seal_s,
            "compression": {"owners": [record["metrics"] for record in records]},
        }
        report["rounds"].append(row)
        _save(args.output / "fixture.json", report)
        print(json.dumps({key: value for key, value in row.items() if key != "publications"}), flush=True)


async def _request(client, endpoint, payload=None, timeout=1200):
    async with httpx.AsyncClient(trust_env=False, timeout=timeout) as http:
        response = await (
            http.get(client.server_url + "/" + endpoint)
            if payload is None
            else http.post(client.server_url + "/" + endpoint, json=payload)
        )
        response.raise_for_status()
        return response.json() if response.content else None


async def _ready(client, process, timeout):
    async def poll():
        while True:
            if process.poll() is not None:
                raise RuntimeError("Engine exited during startup; inspect retained server log")
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=2) as http:
                    response = await http.get(client.server_url + "/health")
                    if response.status_code == 200:
                        return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _freeze_gc(client):
    async with httpx.AsyncClient(trust_env=False, timeout=60) as http:
        response = await http.post(client.server_url + "/freeze_gc")
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
    # Keep fixture creation independent of serving client imports.
    from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient

    processes, logs, clients = [], [], []
    commands = []
    try:
        specs = _engine_specs(args.ports)
        for engine, spec in enumerate(specs):
            port = spec["port"]
            with socket.socket() as sock:
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
            clients.append(SGLangApiClient(f"http://127.0.0.1:{port}"))
            commands.append(spec | {"command": command, "pid": process.pid})
        _save(
            args.output / "launch.json",
            {
                "engines": commands,
                "feature_env": {
                    key: os.environ.get(key)
                    for key in (
                        "GPU_DELTA_TIMING",
                        "GPU_DELTA_SKIP_PAYLOAD_HASH",
                        "GPU_DELTA_DECODE_STAGES",
                        "GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS",
                        "GPU_DELTA_LAYERS_PER_BATCH",
                        "GPU_DELTA_CPU_WORKERS",
                        "GPU_DELTA_HOST_CACHE_DIR",
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
        # Only the benchmark's own fresh process groups; the devbox is retained.
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
            started = time.monotonic()
            try:
                receipt = await activate_publication(clients, cohort, version["publications"][codec])
                result = {
                    "version": version["version"],
                    "coordinator_s": time.monotonic() - started,
                    "codec": codec,
                    "fixture_publication": codec,
                    "inner_codec_origin": fixture.get("inner_codec_origin"),
                    "measurement_phase": "first-use-allocation" if version["version"] == 1 else "warm-update",
                    "receipt": receipt,
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
        cleared = await asyncio.gather(*[_request(client, "clear_gpu_delta_state", {}) for client in clients])
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
        cuda = metrics["cuda_event_ms"] if result["timing_enabled"] else None
        rows.append(
            {
                "identity": identities[rank],
                "outer_cpu_s": _seconds(metrics["host_rank_outer_zstd_decode_s"]),
                "plain_copy_s": _seconds(metrics["host_rank_encoded_copy_s"]),
                "de_stream_cuda_s": _seconds(cuda["decode"]) / 1000 if cuda is not None else None,
                "matrix_apply_cuda_s": _seconds(cuda["layout_apply"]) / 1000 if cuda is not None else None,
                "prepare_s": _seconds(metrics["host_prepare_s"]),
                "pause_s": pause,
                "skip_payload_hash": metrics["host_encoded_cache_skip_payload_hash"],
                "receipt": resumed,
            }
        )
    return sorted(rows, key=lambda row: (row["identity"]["engine_id"], row["identity"]["dp_rank"]))


def _owner_compression(metrics, codec):
    enabled, final = metrics["timing_enabled"], metrics["finalization"]
    inner = _seconds(metrics["batch_cuda_s"]["compression_s"]) if enabled else None
    outer = None
    if codec == "lz4":
        outer = 0.0
    elif enabled:
        outer = _seconds(final["finalize_cuda_phase_s"]["outer_zstd_s"])
    return {
        "inner_cuda_s": inner,
        "outer_cuda_s": outer,
        "batch_wall_s": _seconds(metrics["batch_wall_s"]),
        "outer_wall_s": _seconds(final["outer_zstd_wall_s"]),
        "pack_d2h_wall_s": _seconds(final["pack_d2h_wall_s"]),
        "encode_and_target_write_s": _seconds(metrics["encode_and_target_write_s"]),
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
                "inner_cuda_s",
                "outer_cuda_s",
                "batch_wall_s",
                "outer_wall_s",
                "pack_d2h_wall_s",
                "encode_and_target_write_s",
            )
        },
        "seal_s": _seconds(row["seal_s"]),
        "owners": owners,
    }


def _maximum(rows, key):
    values = [row[key] for row in rows]
    return None if any(value is None for value in values) else max(values)


def _format(value):
    return "unmeasured" if value is None else f"{value:.9f}"


def _markdown(summary):
    lines = [
        "# GPU delta end-to-end benchmark",
        "",
        f"Codec: `{summary['codec']}`. Final selected outputs match the independently loaded target checkpoint on "
        f"all {len(summary['comparison']['routes'])} routes, including text, token IDs and input/output logprobs.",
        "",
        f"Sender owners: {summary['sender_count']}. Layer-ordered assignments and all per-owner metrics are retained in `summary.json`.",
        "",
        "Versions are cumulative synthetic targets, not learned updates or statistical repeats. "
        "Selected outputs do not prove every weight byte. Startup and oracle generation are outside update timing.",
        "",
        "Compression columns are independent owner maxima. Inner CUDA is the maximum of each owner's summed batch events; "
        "these maxima can come from different owners and are not a synchronized end-to-end latency. "
        "CUDA events cover their named stream regions, including wrapper/launch gaps, not pure kernel busy time. "
        "Batch wall time includes pinned-input transfers, "
        "XOR and metadata waits; packing/D2H is separate. Owner total also includes checkpoint reads/writes, "
        "payload hashing and shard publication writing. Parent sealing is measured separately after all owners finish. "
        "These overlapping scopes must not be added.",
        "",
        "|Version|Inner CUDA max s|Outer CUDA max s|Batch wall max s|Outer wall max s|Pack/D2H wall max s|Owner total max s|Parent seal s|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    compression_keys = (
        "inner_cuda_s",
        "outer_cuda_s",
        "batch_wall_s",
        "outer_wall_s",
        "pack_d2h_wall_s",
        "encode_and_target_write_s",
    )
    for row in summary["versions"]:
        lines.append(
            f"|{row['version']}|"
            + "|".join(_format(row["compression"]["owner_max"][key]) for key in compression_keys)
            + f"|{_format(row['compression']['seal_s'])}|"
        )
    lines += [
        "",
        "Receiver columns are independent maxima over ranks, not rank sums or additive phases. "
        "Outer CPU/plain-copy wall includes raw copies and job submission/drain; worker sums remain in the raw receipts. "
        "DE stream events include zero-fill and nvCOMP enqueue/host gaps; they are not pure hardware busy time. "
        "Matrix apply includes status checks but excludes raw copies and derived refresh. "
        "Pause comes from completed scheduler pause/resume timestamps. "
        "Disabled CUDA timing is unmeasured; plain LZ4 has no outer stage.",
        "",
        "|Version|Outer CPU max s|Plain copy max s|DE stream CUDA max s|Matrix apply CUDA max s|Prepare max s|Pause max s|Coordinator s|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rank_keys = ("outer_cpu_s", "plain_copy_s", "de_stream_cuda_s", "matrix_apply_cuda_s", "prepare_s", "pause_s")
    for row in summary["versions"]:
        values = [_format(row["receiver_max"][key]) for key in rank_keys] + [_format(row["coordinator_s"])]
        lines.append(f"|{row['version']}|" + "|".join(values) + "|")
    lines += [
        "",
        "|Version|Sender payload checksum|Receiver skips payload SHA|Inner frame bytes|Canonical bytes|Changed bytes|Inner compressed bytes|Matrix payload bytes|Payload file bytes|Manifest bytes|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["versions"]:
        values = [
            row["sender_payload_checksum_format"],
            row["receiver_skip_payload_hash"],
            row["frame_bytes"],
            row["canonical_bytes"],
            row["changed_bytes"],
        ]
        values += [
            row["accounting"][key]
            for key in ("encoded_frame_bytes", "matrix_payload_bytes", "payload_file_bytes", "manifest_bytes")
        ]
        lines.append(f"|{row['version']}|" + "|".join(map(str, values)) + "|")
    lines += [
        "",
        "## Raw owner timings",
        "",
        "|Version|Owner|GPU|Canonical bytes|Inner CUDA s|Outer CUDA s|Batch wall s|Outer wall s|Pack/D2H wall s|Owner total s|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for version in summary["versions"]:
        for owner in version["compression"]["owners"]:
            metrics = owner["raw_metrics"]
            lines.append(
                f"|{version['version']}|{metrics['owner']}|{metrics['gpu']}|{metrics['canonical_bytes']}|"
                + "|".join(_format(owner[key]) for key in compression_keys)
                + "|"
            )
    lines += [
        "",
        "## Raw rank timings",
        "",
        "Full receipts and encoder metrics are retained in `summary.json`.",
        "",
        "|Version|Engine|DP rank|Rank ID|Outer CPU s|Plain copy s|DE stream CUDA s|Matrix apply CUDA s|Prepare s|Pause s|",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for version in summary["versions"]:
        for row in version["ranks"]:
            identity = row["identity"]
            lines.append(
                f"|{version['version']}|{identity['engine_id']}|{identity['dp_rank']}|{identity['rank_id']}|"
                + "|".join(_format(row[key]) for key in rank_keys)
                + "|"
            )
    return "\n".join(lines) + "\n"


def _write_report(output: Path):
    fixture = _read(output / "fixture" / "fixture.json")
    codec = fixture["codec"]
    sender_owners = fixture["sender_owners"]
    assignments = {owner["owner"]: owner for owner in sender_owners}
    if len(assignments) != len(sender_owners) or len(assignments) != fixture["sender_gpus"]:
        raise ValueError("Sender owner plan differs from the fixture")
    inventory = _read(output / "receiver" / "inventory.json")
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
        update = _read(output / "receiver" / f"update-{version}.json")
        if update["version"] != version or update["codec"] != codec:
            raise ValueError("Update version/codec differs")
        ranks = _rank_rows(update, publication, identities)
        hash_policies = {rank["skip_payload_hash"] for rank in ranks}
        if len(hash_policies) != 1:
            raise ValueError("Receiver ranks have different payload hash policies")
        selected = _generations(_read(output / "receiver" / f"generation-{version}.json"), routes, version)
        versions.append(
            {
                "version": version,
                "frame_bytes": publication["frame_bytes"],
                "sender_payload_checksum_format": publication["payload_checksum_format"],
                "receiver_skip_payload_hash": hash_policies.pop(),
                "canonical_bytes": row["canonical_bytes"],
                "changed_bytes": row["changed_bytes"],
                "accounting": row["accounting"][codec],
                "compression": _compression(row, codec, assignments),
                "coordinator_s": _seconds(update["coordinator_s"]),
                "receiver_max": {
                    key: _maximum(ranks, key)
                    for key in (
                        "outer_cpu_s",
                        "plain_copy_s",
                        "de_stream_cuda_s",
                        "matrix_apply_cuda_s",
                        "prepare_s",
                        "pause_s",
                    )
                },
                "ranks": ranks,
            }
        )
    if not versions:
        raise ValueError("No fixture updates")
    oracle = _generations(_read(output / "oracle" / "target-generation.json"), routes)
    comparison = _compare(selected, oracle)
    _save(output / "comparison.json", comparison)
    if comparison["status"] != "PASS":
        raise ValueError("Final selected generation differs from target checkpoint; see comparison.json")
    summary = {
        "status": "PASS",
        "codec": codec,
        "target_checkpoint": fixture["target_checkpoint"],
        "sender_count": fixture["sender_gpus"],
        "sender_owners": sender_owners,
        "comparison": comparison,
        "versions": versions,
    }
    report = _markdown(summary)
    _save(output / "summary.json", summary)
    (output / "REPORT.md").write_text(report)
    return summary


def _phase(phase, args, directory, settings):
    os.setsid()  # The parent forwards Ctrl-C once while this process drains its workers.
    # Process exit releases each phase's CUDA context before the next phase starts.
    os.environ.update(settings)
    with (args.output / f"{directory}.log").open("x") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        stage = argparse.Namespace(
            **(
                vars(args)
                | {
                    "output": args.output / directory,
                    "inventory": args.output / "inventory" / "inventory.json",
                    "fixture": args.output / "fixture" if phase in {"run", "oracle"} else None,
                }
            )
        )
        stage.output.mkdir()
        if phase == "fixture":
            _fixture(stage)
        else:
            asyncio.run(_run(stage, phase))


def _end_to_end(args):
    settings = {
        "GPU_DELTA_CODEC": configured_codec(),
        "GPU_DELTA_TIMING": "1",  # This workload reports instrumented CUDA phases.
        "GPU_DELTA_SKIP_PAYLOAD_HASH": "0" if args.verify_payload_hash else "1",
    }
    records = []
    for phase, directory in [
        ("inventory", "inventory"),
        ("fixture", "fixture"),
        ("run", "receiver"),
        ("oracle", "oracle"),
    ]:
        # Fixtures retain hashes; receiver verification is the measured policy.
        phase_settings = settings | ({"GPU_DELTA_SKIP_PAYLOAD_HASH": "0"} if phase == "fixture" else {})
        process = multiprocessing.get_context("spawn").Process(
            target=_phase, args=(phase, args, directory, phase_settings), name=phase
        )
        record = {"phase": phase, "environment": phase_settings, "status": "RUNNING"}
        records.append(record)
        _save(args.output / "workflow.json", {"phases": records})
        print(f"{phase}: {args.output / directory}", flush=True)
        started = time.monotonic()
        process.start()
        record["pid"] = process.pid
        _save(args.output / "workflow.json", {"phases": records})
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
        _save(args.output / "workflow.json", {"phases": records})
        if process.exitcode != 0:
            raise RuntimeError(f"{phase} failed; see {args.output / (directory + '.log')}")
    _write_report(args.output)
    print((args.output / "REPORT.md").read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Existing NVFP4 HF serving checkpoint")
    parser.add_argument("--output", type=Path, help="New output directory; generated automatically if omitted")
    parser.add_argument("--codec", choices=CODECS, help="Override GPU_DELTA_CODEC (default: snappy-zstd)")
    parser.add_argument("--frame-bytes", type=int, default=FRAME_BYTES, help="Inner frame bytes; at most 4 MiB")
    parser.add_argument("--sender-gpus", type=int, help="Compression GPUs; defaults to all visible GPUs")
    parser.add_argument("--verify-payload-hash", action="store_true", help="Enable receiver payload SHA verification")
    parser.add_argument("--versions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument(
        "--ratio", type=float, default=0.002, help="CPU Zstd calibration target for synthetic mutation rate"
    )
    parser.add_argument(
        "--ports", type=int, nargs="+", default=[31000, 31100], help="Two ports: two EP4 engines; one port: EP8"
    )
    parser.add_argument("--startup-timeout", type=float, default=3600)
    args = parser.parse_args()
    args.model = args.model.resolve(strict=True)
    args.output = (
        args.output or Path("gpu-delta-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
    ).resolve()
    if args.codec is not None:
        os.environ["GPU_DELTA_CODEC"] = args.codec
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
    args.output.mkdir(parents=True, exist_ok=False)
    _save(args.output / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    _end_to_end(args)


if __name__ == "__main__":
    main()
