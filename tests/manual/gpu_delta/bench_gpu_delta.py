"""One EP8 or two EP4 GLM5.2 W4A16 engines: GPU delta fixture and timing harness.

Requires paired SGLang GPU-delta sources, eight Blackwell GPUs, the Miles image,
prebuilt nvCOMP >=5.3 and an immutable NVFP4 checkpoint with bundled MTP weights.
No private model names/paths or cluster launch commands are embedded here.

Example (run from the Miles checkout with paired SGLang on PYTHONPATH)::

    python tests/manual/gpu_delta/bench_gpu_delta.py inventory --model /models/base --output /data/inventory
    python tests/manual/gpu_delta/bench_gpu_delta.py fixture --model /models/base \
        --inventory /data/inventory/inventory.json --output /data/fixture
    GPU_DELTA_CODEC=snappy-zstd GPU_DELTA_TIMING=1 \
        python tests/manual/gpu_delta/bench_gpu_delta.py run --model /models/base \
        --fixture /data/fixture --output /data/snappy-zstd

Canonical rank-zero/rank-one tensors use direct uncompressed target values when
changed; all other tensors retain compressed XOR frames. Fixture accounting keeps
direct-value traffic separate. Each run starts fresh engines (one port selects
EP8; two ports select two EP4 engines on disjoint four-GPU slices);
never reset a live delta stream with a disk reload. ``oracle`` starts from the
altered target checkpoint while keeping the original static draft, outside update
timing. Compare its generation/logprob records with the last delta round.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from contextlib import asynccontextmanager
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
    DTYPE_BYTES,
    FRAME_BYTES,
    PublicationWriter,
    canonical_json,
    configured_codec,
    seal_publication,
    sha256,
    tensor_metadata,
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
    "cuda_graph_max_bs_decode": 256,
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


def _validate_fixture_codec(fixture, codec):
    if fixture.get("codec") != codec:
        raise ValueError(f"Fixture codec differs from configured {codec}")
    for row in fixture["rounds"]:
        publication = row["publications"].get(codec)
        if (
            publication is None
            or publication.get("protocol_version") != 4
            or publication.get("codec") != codec
            or type(publication.get("frame_bytes")) is not int
            or publication["frame_bytes"] not in (1 << 16, 1 << 19, FRAME_BYTES, 1 << 22)
        ):
            raise ValueError(f"Fixture requires protocol 4 / {codec} / 64 KiB, 512 KiB, 1 MiB or 4 MiB inner frames")
    return codec


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


def _payload_links(manifest, source, destination):
    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    files = []
    for item in manifest["files"]:
        path = source / item["name"]
        if path.name != item["name"] or path.is_symlink() or path.resolve(strict=True).parent != source:
            raise ValueError("Payload must be an ordinary file inside the source publication")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while chunk := stream.read(8 * 1024**2):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        if (
            identity(before) != identity(after)
            or after.st_size != item["nbytes"]
            or digest.hexdigest() != item["sha256"]
        ):
            raise ValueError("Source payload size, bytes or identity changed")
        target = destination / item["name"]
        # EXDEV is an explicit setup failure; never copy/re-encode payloads or
        # follow symlinks as a fallback. Hardlinks preserve the verified bytes.
        os.link(path, target, follow_symlinks=False)
        if identity(target.stat()) != identity(after):
            raise ValueError("Rebound payload does not share the verified source inode")
        files.append(
            item | {"source": str(path), "target": str(target), "device": after.st_dev, "inode": after.st_ino}
        )
    return files


def _rebind_round(args, row, plan, source_digest, target_digest, stream_id, codec):
    publication = row["publications"][codec]
    path = Path(publication["manifest_path"]).resolve(strict=True)
    content = path.read_bytes()
    if sha256(content) != publication["manifest_sha256"]:
        raise ValueError("Source manifest SHA256 differs from the immutable descriptor")
    manifest = json.loads(content)
    metadata = {key: value for key, value in manifest.items() if key not in {"tensors", "files"}}
    if any(
        metadata.get(key) != value
        for key, value in publication.items()
        if key not in {"manifest_path", "manifest_sha256"}
    ):
        raise ValueError("Source publication metadata differs from its descriptor")
    old_plan = [
        {key: tensor[key] for key in ("name", "dtype", "shape", "encoding", "views")} for tensor in manifest["tensors"]
    ]
    if sha256(canonical_json(old_plan)) != source_digest or metadata["plan_digest"] != source_digest:
        raise ValueError("Source fixture canonical plan digest differs")
    if [tensor["name"] for tensor in manifest["tensors"]] != [tensor["name"] for tensor in plan]:
        raise ValueError("Source and target canonical tensor inventories differ")
    tensors = []
    for old, new in zip(manifest["tensors"], plan, strict=True):
        expected = tensor_metadata(**new)
        if any(old[key] != expected[key] for key in ("name", "dtype", "shape", "encoding", "nbytes", "byte_order")):
            raise ValueError(f"Canonical tensor metadata differs: {old['name']}")
        tensors.append(old | {"views": expected["views"]})
    invariant_digest = sha256(
        canonical_json([{key: value for key, value in tensor.items() if key != "views"} for tensor in tensors])
    )
    if invariant_digest != sha256(
        canonical_json(
            [{key: value for key, value in tensor.items() if key != "views"} for tensor in manifest["tensors"]]
        )
    ):
        raise ValueError("Rebinding changed canonical payload/frame metadata")
    directory = args.output / codec / f"v{row['version']}"
    directory.mkdir(parents=True, exist_ok=False)
    files = _payload_links(manifest, path.parent, directory)
    metadata |= {
        "plan_digest": target_digest,
        "stream_id": stream_id,
        "publication_id": f"{stream_id}:{row['version']}",
    }
    rebound = seal_publication(directory, [{"metadata": metadata, "files": manifest["files"], "tensors": tensors}])
    sizes = _publication_accounting(rebound)
    result = row | {
        "publications": {codec: rebound},
        "accounting": {codec: sizes},
        "ratios": {codec: sizes["publication_bytes"] / row["canonical_bytes"]},
    }
    proof = {
        "version": row["version"],
        "source_publication": publication,
        "target_publication": rebound,
        "tensor_count": len(tensors),
        "non_view_tensor_metadata_sha256": invariant_digest,
        "files": files,
        "payload_bytes": sizes["payload_file_bytes"],
    }
    return result, proof


def _rebind(args):
    """Rebind only canonical views to a fresh inventory, without changing weights."""
    source_path = (args.fixture / "fixture.json").resolve(strict=True)
    source_bytes = source_path.read_bytes()
    fixture = json.loads(source_bytes)
    codec = configured_codec()
    _validate_fixture_codec(fixture, codec)
    inventory_bytes = args.inventory.read_bytes()
    inventory = json.loads(inventory_bytes)
    cohort = negotiate_cohort(inventory["descriptions"])
    _validate_cohort(cohort, len(inventory["descriptions"]))
    if inventory["plan_digest"] != cohort.plan_digest:
        raise ValueError("Saved inventory plan digest differs from its actual participants")
    index = _tensor_index(args.model)
    for tensor in cohort.plan:
        actual = index[tensor["name"]]
        if tensor["shape"] != actual["shape"] or tensor["dtype"] != actual["dtype"]:
            raise ValueError(f"Inventory/checkpoint mismatch: {tensor['name']}")
    stream_id = sha256(
        canonical_json(
            {
                "source_fixture_sha256": sha256(source_bytes),
                "plan_digest": cohort.plan_digest,
                "output": str(args.output.resolve()),
            }
        )
    )
    report = fixture | {
        "plan_digest": cohort.plan_digest,
        "stream_id": stream_id,
        "rounds": [],
        "topology_rebind": "rebind.json",
    }
    proof = {
        "source_fixture": str(source_path),
        "source_fixture_sha256": sha256(source_bytes),
        "inventory": str(args.inventory.resolve()),
        "inventory_sha256": sha256(inventory_bytes),
        "source_plan_digest": fixture["plan_digest"],
        "target_plan_digest": cohort.plan_digest,
        "source_stream_id": fixture["stream_id"],
        "target_stream_id": stream_id,
        "target_checkpoint": fixture["target_checkpoint"],
        "engine_ids": list(cohort.engine_ids),
        "participants": list(cohort.identities),
        "rounds": [],
    }
    for version, row in enumerate(fixture["rounds"], 1):
        publication = row["publications"][codec]
        if (
            row["version"] != version
            or publication["base_version"] != version - 1
            or publication["target_version"] != version
            or publication["stream_id"] != fixture["stream_id"]
        ):
            raise ValueError("Fixture publications must be one consecutive cumulative stream")
        result, round_proof = _rebind_round(
            args, row, cohort.plan, fixture["plan_digest"], cohort.plan_digest, stream_id, codec
        )
        report["rounds"].append(result)
        proof["rounds"].append(round_proof)
    _save(args.output / "fixture.json", report)
    proof |= {
        "status": "PASS",
        "fixture_sha256": sha256((args.output / "fixture.json").read_bytes()),
        "scope": "Views/plan/stream/publication identities rebound; all non-view tensor metadata and payload bytes unchanged. Original final target checkpoint retained.",
    }
    _save(args.output / "rebind.json", proof)


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


def _fixture(args):
    import torch

    from miles.utils.gpu_delta.encoder import GpuBatchEncoder

    codec = configured_codec()
    inventory = json.loads(args.inventory.read_text())
    plan, _, digest = merge_plans(inventory["descriptions"])
    index = _tensor_index(args.model)
    for spec in plan:
        actual = index[spec["name"]]
        if spec["shape"] != actual["shape"] or spec["dtype"] != actual["dtype"]:
            raise ValueError(f"Inventory/checkpoint mismatch: {spec['name']}")
    target = args.output / "altered-checkpoint"
    target.mkdir()
    # An independent copy; writing cumulative targets never mutates the source.
    for path in args.model.iterdir():
        if path.is_file() and (path.suffix in {".json", ".py", ".model", ".tiktoken", ".safetensors"}):
            shutil.copy2(path, target / path.name)
    rate = _calibrate(plan, args.seed, args.ratio)
    stream_id = sha256(canonical_json({"seed": args.seed, "plan": digest, "rate": rate}))
    report = {
        "plan_digest": digest,
        "rate": rate,
        "requested_zstd_ratio": args.ratio,
        "stream_id": stream_id,
        "target_checkpoint": str(target.resolve()),
        "rounds": [],
        "calibration": "Plain CPU Zstd level-1 sample-frame estimate; final selected-codec size is measured separately, not forced to the requested ratio.",
        "canonical_denominator": "Mutable canonical tensors in the receiver plan; excludes frozen draft and non-updated checkpoint entries.",
        "changed_bytes_definition": "Unequal storage bytes, not changed bits or compressed size.",
        "codec": codec,
        "inner_codec_origin": f"Production GpuBatchEncoder: pinned snapshots, GPU XOR/{codec.removesuffix('-zstd')}, "
        + ("no outer compression" if codec == "lz4" else "then one GPU Zstd submission per version")
        + ". Fixture setup is excluded from receiver timing.",
    }
    encoder = GpuBatchEncoder(torch.device("cuda", torch.cuda.current_device()), codec=codec)
    denominator = sum(index[t["name"]]["nbytes"] for t in plan)
    raw_plan = [t for t in plan if t["encoding"] == "raw_bytes"]
    matrix_plan = [t for t in plan if t["encoding"] == "xor_bytes"]
    if len(raw_plan) + len(matrix_plan) != len(plan):
        raise ValueError("Unsupported fixture tensor encoding")

    for version in range(1, args.versions + 1):
        writer = PublicationWriter(
            args.output / codec / f"v{version}",
            stream_id=stream_id,
            base_version=version - 1,
            target_version=version,
            plan_digest=digest,
            publication_id=f"{stream_id}:{version}",
            codec=codec,
        )
        started, changed = time.monotonic(), 0
        pending, batch, batch_bytes = [], [], 0
        try:
            for tensor in raw_plan:
                before, after = _alter_fixture_tensor(
                    target, index, tensor, version=version, seed=args.seed, rate=rate
                )
                entry = writer.add_raw_tensor(
                    tensor["name"], before, after, dtype=tensor["dtype"], shape=tensor["shape"], views=tensor["views"]
                )
                changed += entry["changed_bytes"]
            for tensor in matrix_plan:
                before, after = _alter_fixture_tensor(
                    target, index, tensor, version=version, seed=args.seed, rate=rate
                )
                # Pinned snapshots are bounded by a batching target, except that
                # one larger tensor remains whole. Only compact inner payloads survive
                # each GPU batch; the canonical snapshots are then released.
                batch.append((torch.from_numpy(before.copy()).pin_memory(), torch.from_numpy(after).pin_memory()))
                batch_bytes += after.nbytes
                if batch_bytes >= 512 * 1024**2:
                    pending.extend(encoder.encode_device(batch))
                    batch, batch_bytes = [], 0
            if batch:
                pending.extend(encoder.encode_device(batch))
            batch = []
            for tensor, (frames, payload, outer, count, _) in zip(
                matrix_plan, encoder.finish_device(pending), strict=True
            ):
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
            publication = writer.finish()
        finally:
            writer.close()
        sizes = _publication_accounting(publication)
        row = {
            "version": version,
            "publications": {codec: publication},
            "canonical_bytes": denominator,
            "changed_bytes": changed,
            "changed_byte_fraction": changed / denominator,
            "accounting": {codec: sizes},
            "ratios": {codec: sizes["publication_bytes"] / denominator},
            "encoded_frame_ratios": {codec: sizes["encoded_frame_bytes"] / denominator},
            "encode_and_target_write_s": time.monotonic() - started,
        }
        report["rounds"].append(row)
        _save(args.output / "fixture.json", report)
        print(json.dumps({k: v for k, v in row.items() if k != "publications"}), flush=True)


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


@asynccontextmanager
async def _engines(args, model, codec):
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
            env = os.environ | SERVER_ENV | {"CUDA_VISIBLE_DEVICES": spec["gpu_ids"], "GPU_DELTA_CODEC": codec}
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
                    key: codec if key == "GPU_DELTA_CODEC" else os.environ.get(key)
                    for key in (
                        "GPU_DELTA_CODEC",
                        "GPU_DELTA_TIMING",
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
        yield clients
    finally:
        # Only the benchmark's own fresh process groups; the devbox is retained.
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                await asyncio.to_thread(process.wait, timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.to_thread(process.wait)
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
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        }
        values = await asyncio.gather(*[_request(client, "generate", payload) for client in clients])
        records.append({"dp_rank": dp_rank, "engine_ids": engine_ids, "engines": values})
    return records


async def _run(args):
    codec = configured_codec()
    fixture_key = codec
    fixture = json.loads((args.fixture / "fixture.json").read_text()) if args.fixture else None
    if args.phase == "run":
        fixture_key = _validate_fixture_codec(fixture, codec)
    model = Path(fixture["target_checkpoint"]) if args.phase == "oracle" else args.model
    async with _engines(args, model, codec) as clients:
        if args.phase == "oracle":
            _save(args.output / "target-generation.json", await _generation(clients))
            return
        descriptions = await asyncio.gather(
            *[c.get_weights_delta_info(engine_id=f"engine-{i:05d}") for i, c in enumerate(clients)]
        )
        cohort = negotiate_cohort(descriptions)
        digest = cohort.plan_digest
        _validate_cohort(cohort, len(clients))
        _save(args.output / "gpu-processes.json", _capture_gpu_processes(cohort, args.ports))
        _save(args.output / "inventory.json", {"descriptions": descriptions, "plan_digest": digest})
        if args.phase == "inventory":
            return
        if digest != fixture["plan_digest"]:
            raise ValueError("Fixture plan no longer matches current loaded models")
        _save(args.output / "base-generation.json", await _generation(clients))
        for version in fixture["rounds"]:
            started = time.monotonic()
            try:
                receipt = await activate_publication(clients, cohort, version["publications"][fixture_key])
                result = {
                    "version": version["version"],
                    "coordinator_s": time.monotonic() - started,
                    "codec": codec,
                    "fixture_publication": fixture_key,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("inventory", "fixture", "rebind", "run", "oracle"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--versions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--ratio", type=float, default=0.002)
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=(31000,),
        help="One port: one EP8 engine; two ports: two EP4 engines on GPUs 0-3 and 4-7",
    )
    parser.add_argument("--startup-timeout", type=float, default=3600)
    args = parser.parse_args()
    args.model = args.model.resolve(strict=True)
    if args.phase in {"fixture", "rebind"} and args.inventory is None:
        parser.error("fixture/rebind requires --inventory")
    if args.phase in {"rebind", "run", "oracle"} and args.fixture is None:
        parser.error("rebind/run/oracle requires --fixture")
    try:
        _engine_specs(args.ports)
    except ValueError as error:
        parser.error(str(error))
    if args.versions < 1 or not 0 < args.ratio < 0.1:
        parser.error("versions must be positive and ratio in (0, 0.1)")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.phase == "fixture":
        _fixture(args)
    elif args.phase == "rebind":
        _rebind(args)
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
