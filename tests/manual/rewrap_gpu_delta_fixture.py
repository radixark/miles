"""Rewrap a retained CPU-outer Snappy fixture with GPU Zstd, preserving targets.

Setup only: authenticate the immutable source publications, CPU-decode their old
outer envelopes into pinned tensors, and pass every tensor arena together to the
production GPU outer encoder. No canonical matrix is decoded, hashed, mutated or
recompressed with Snappy; the altered checkpoint is reused unchanged. Exact
inner arena and raw target hashes are checked after writing the new publications.
This setup is excluded from sender/receiver benchmark update timings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import zstandard

from miles.utils.gpu_delta_encoder import DeviceEncodedTensor, GpuBatchEncoder
from miles.utils.gpu_delta_publication import PublicationWriter, canonical_json


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("xb") as f:
        f.write(canonical_json(value))
        f.flush()
        os.fsync(f.fileno())


def read_range(files, record):
    f = files[record["file"]]
    f.seek(record["encoded_offset"])
    result = f.read(record["encoded_bytes"])
    if len(result) != record["encoded_bytes"]:
        raise ValueError("Short immutable publication range")
    return result


def source_files(manifest_path, manifest):
    files = {}
    try:
        for entry in manifest["files"]:
            name = entry["name"]
            path = manifest_path.parent / name
            if Path(name).name != name or path.is_symlink() or path.stat().st_size != entry["nbytes"] or file_digest(path) != entry["sha256"]:
                raise ValueError("Source payload path, size or digest differs")
            files[name] = path.open("rb")
        return files
    except BaseException:
        for f in files.values():
            f.close()
        raise


def verify_publication(descriptor, expected):
    path = Path(descriptor["manifest_path"])
    if file_digest(path) != descriptor["manifest_sha256"]:
        raise ValueError("New manifest digest differs")
    manifest = json.loads(path.read_text())
    if {e["name"] for e in manifest["tensors"]} != set(expected):
        raise ValueError("Rewrapping changed the canonical inventory")
    files = source_files(path, manifest)
    proof = []
    try:
        for entry in manifest["tensors"]:
            previous = expected[entry["name"]]
            stable = {k: entry[k] for k in ("name", "dtype", "shape", "encoding", "views", "nbytes", "changed_bytes")}
            if stable != previous["canonical_metadata"]:
                raise ValueError("Rewrapping changed canonical metadata")
            if entry["encoding"] == "raw_bytes":
                data = read_range(files, entry["raw"]) if "raw" in entry else b""
                if digest(data) != previous["payload_sha256"] or len(data) != previous["bytes"]:
                    raise ValueError("Raw target bytes changed")
            elif entry["frames"]:
                outer = entry["outer"]
                encoded = read_range(files, outer)
                arena = bytearray(outer["decoded_bytes"])
                decoded_end = 0
                for frame in outer["frames"]:
                    if frame["decoded_offset"] != decoded_end:
                        raise ValueError("Outer chunks do not cover the inner arena")
                    start = frame["encoded_offset"]
                    block = zstandard.ZstdDecompressor().decompress(
                        encoded[start:start + frame["encoded_bytes"]], max_output_size=frame["decoded_bytes"]
                    )
                    if len(block) != frame["decoded_bytes"]:
                        raise ValueError("Outer decoded size differs")
                    arena[decoded_end:decoded_end + len(block)] = block
                    decoded_end += len(block)
                if decoded_end != len(arena) or digest(arena) != previous["payload_sha256"]:
                    raise ValueError("GPU rewrapped Snappy arena changed")
                geometries = [{k: v for k, v in f.items() if k not in ("file", "encoded_sha256")} for f in entry["frames"]]
                if geometries != previous["inner_frames"]:
                    raise ValueError("Inner Snappy frame geometry changed")
            elif previous["bytes"] != 0:
                raise ValueError("Changed matrix lost its inner frames")
            proof.append({"name": entry["name"], "encoding": entry["encoding"], "bytes": previous["bytes"], "payload_sha256": previous["payload_sha256"]})
    finally:
        for f in files.values():
            f.close()
    return manifest, proof


def rewrap(args):
    started = time.monotonic()
    source_raw = args.fixture.read_bytes()
    if digest(source_raw) != args.fixture_sha256:
        raise ValueError("Source fixture digest differs")
    fixture = json.loads(source_raw)
    if fixture["codecs"] != ["snappy-zstd"] or [r["version"] for r in fixture["rounds"]] != [1, 2, 3]:
        raise ValueError("Expected the retained three-version CPU-outer fixture")
    args.output.mkdir()  # Exclusive: retain failed/partial setup rather than overwrite.
    derived = copy.deepcopy(fixture)
    key = "snappy-gpu-zstd"
    derived.update(codecs=[key], rounds=[])
    derived["gpu_outer_derivation"] = {
        "source_fixture": str(args.fixture.resolve()), "source_fixture_sha256": args.fixture_sha256,
        "source_digest": args.source_digest, "setup_only": True,
        "contract": "Exact retained Snappy arenas and raw targets; GPU outer Zstd framing only, no new mutation or recalibration.",
    }
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    encoder = GpuBatchEncoder("snappy", device, outer_backend="gpu")
    report = {"status": "INCOMPLETE", **derived["gpu_outer_derivation"], "rounds": []}
    for original_row in fixture["rounds"]:
        version = original_row["version"]
        descriptor = original_row["publications"]["snappy-zstd"]
        path = Path(descriptor["manifest_path"])
        raw = path.read_bytes()
        if digest(raw) != descriptor["manifest_sha256"]:
            raise ValueError("Original manifest checksum differs")
        manifest = json.loads(raw)
        if (manifest["protocol_version"] != 3 or manifest["codec_profile"] != "snappy-independent-1mib-zstd-v1"
                or manifest["plan_digest"] != fixture["plan_digest"]):
            raise ValueError("Expected immutable protocol3 CPU outer Zstd source")
        files = source_files(path, manifest)
        pending, specs, pinned, expected, raw_targets = [], [], [], {}, []
        try:
            for entry in manifest["tensors"]:
                metadata = {k: entry[k] for k in ("name", "dtype", "shape", "encoding", "views", "nbytes", "changed_bytes")}
                if entry["encoding"] == "raw_bytes":
                    payload = read_range(files, entry["raw"]) if "raw" in entry else b""
                    raw_targets.append((entry, payload))
                    expected[entry["name"]] = {"canonical_metadata": metadata, "bytes": len(payload), "payload_sha256": digest(payload)}
                    continue
                if entry["encoding"] != "xor_bytes":
                    raise ValueError("Unexpected matrix encoding")
                frames = [{k: v for k, v in f.items() if k not in ("file", "encoded_sha256")} for f in entry["frames"]]
                arena, payload = b"", None
                if frames:
                    outer = entry["outer"]
                    arena = zstandard.ZstdDecompressor().decompress(read_range(files, outer), max_output_size=outer["decoded_bytes"])
                    if len(arena) != outer["decoded_bytes"]:
                        raise ValueError("CPU outer decoded length differs")
                    for frame in entry["frames"]:
                        start = frame["encoded_offset"]
                        if digest(arena[start:start + frame["encoded_bytes"]]) != frame["encoded_sha256"]:
                            raise ValueError("Original Snappy frame checksum differs")
                    host = torch.empty(len(arena), dtype=torch.uint8, pin_memory=True)
                    host.numpy()[:] = np.frombuffer(arena, dtype=np.uint8)
                    pinned.append(host)
                    with torch.cuda.stream(encoder.stream):
                        payload = host.to(device, non_blocking=True)
                expected[entry["name"]] = {"canonical_metadata": metadata, "bytes": len(arena), "payload_sha256": digest(arena), "inner_frames": frames}
                pending.append(DeviceEncodedTensor(frames, payload, entry["changed_bytes"], {}))
                specs.append(entry)
        finally:
            for f in files.values():
                f.close()
        # A single outer compression submission sees all tensor arenas. Input
        # uploads precede it on the same stream; pinned owners remain alive.
        wrapped = encoder.wrap_device(pending)
        directory = args.output / key / f"v{version}"
        directory.mkdir(parents=True)
        writer = PublicationWriter(directory, stream_id=manifest["stream_id"], base_version=manifest["base_version"], target_version=version,
                                   plan_digest=manifest["plan_digest"], publication_id=manifest["publication_id"], codec="snappy", outer_backend="gpu")
        try:
            for entry, (frames, payload, outer, changed, _) in zip(specs, wrapped, strict=True):
                writer.add_gpu_outer_tensor(entry["name"], frames, payload, outer, changed_bytes=changed,
                                            dtype=entry["dtype"], shape=entry["shape"], views=entry["views"])
            for original, payload in raw_targets:
                # Benchmark-only immutable copy: preserve authentic changed-byte
                # counts without inventing old canonical values to recompute them.
                entry = copy.deepcopy(original)
                entry.pop("raw", None)
                writer._append_tensor(entry, [memoryview(payload)] if payload else [])
            publication = writer.finish()
        finally:
            writer.close()
        new_manifest, proof = verify_publication(publication, expected)
        proof_path = args.output / f"rewrap-proof-v{version}.json"
        save(proof_path, proof)
        outer_encoded = sum(e.get("outer", {}).get("encoded_bytes", 0) for e in new_manifest["tensors"])
        raw_bytes = sum(e.get("raw", {}).get("encoded_bytes", 0) for e in new_manifest["tensors"])
        payload_bytes = sum(f["nbytes"] for f in new_manifest["files"])
        account = {"outer_encoded_bytes": outer_encoded, "outer_decoded_arena_bytes": sum(e.get("outer", {}).get("decoded_bytes", 0) for e in new_manifest["tensors"]),
                   "outer_frames": sum(len(e.get("outer", {}).get("frames", [])) for e in new_manifest["tensors"]),
                   "encoded_frame_bytes": sum(f["encoded_bytes"] for e in new_manifest["tensors"] for f in e["frames"]),
                   "raw_tensor_count": len(raw_targets), "raw_target_bytes": raw_bytes,
                   "payload_file_bytes": payload_bytes, "alignment_bytes": payload_bytes - outer_encoded - raw_bytes,
                   "manifest_bytes": Path(publication["manifest_path"]).stat().st_size}
        account["publication_bytes"] = payload_bytes + account["manifest_bytes"]
        row = copy.deepcopy(original_row)
        row.update(publications={key: publication}, accounting={key: account},
                   ratios={key: account["publication_bytes"] / row["canonical_bytes"]},
                   encoded_frame_ratios={key: account["encoded_frame_bytes"] / row["canonical_bytes"]})
        derived["rounds"].append(row)
        report["rounds"].append({"version": version, "source_manifest_sha256": descriptor["manifest_sha256"], "publication": publication,
                                 "accounting": account, "proof": {"path": str(proof_path.resolve()), "sha256": file_digest(proof_path)},
                                 "exact_inner_arenas_verified": True, "exact_raw_targets_verified": True,
                                 "setup_outer_metrics": encoder.outer_metrics})
        del wrapped, pending, pinned, raw_targets, expected
    save(args.output / "fixture.json", derived)
    report.update(status="PASS", fixture_sha256=file_digest(args.output / "fixture.json"), setup_wall_s=time.monotonic() - started,
                  plan_digest=fixture["plan_digest"], target_checkpoint=fixture["target_checkpoint"])
    save(args.output / "derivation.json", report)
    print(json.dumps({"status": report["status"], "fixture_sha256": report["fixture_sha256"], "setup_wall_s": report["setup_wall_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    for value in (args.fixture_sha256, args.source_digest):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Exact source and fixture digests are required")
    rewrap(args)


if __name__ == "__main__":
    main()
