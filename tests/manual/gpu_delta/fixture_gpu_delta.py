"""HF-only GPU fixture generation with whole-layer ownership across visible GPUs.

Layers precede embedding, LM head and remaining tensors. Like Miles ordinary
ownership, groups use balanced contiguous slices; no Megatron export is involved.
"""

import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zstandard

from miles.backends.training_utils.weight_update.protocols.gpu_delta.session import merge_plans
from miles.utils.gpu_delta.publication import (
    DTYPE_BYTES,
    FRAME_BYTES,
    PublicationWriter,
    canonical_json,
    configured_codec,
    seal_publication,
    sha256,
)


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
