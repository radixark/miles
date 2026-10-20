"""Read-only admission of saved, SHA-on benchmark publications.

This does not decode payloads or replace production receiver admission. The
receiver still authenticates and consumes each unchanged publication. Prepared
checkpoint content retains the original READY/settings/file-size boundary;
payload-byte verification is separate from an every-weight-byte oracle.
"""

import hashlib
import json
import os
import re
import stat
from pathlib import Path

JSON_LIMIT = 256 << 20  # Full-model manifests contain about 174,000 tensors.
IDENTITY = (
    "stream_id",
    "publication_id",
    "plan_digest",
    "base_version",
    "target_version",
    "codec",
    "frame_bytes",
    "payload_checksum_format",
)
PLAN_FIELDS = ("name", "dtype", "shape", "encoding", "views")
PRODUCER_FILES = (
    "tests/manual/gpu_delta/bench_gpu_delta.py",
    "miles/utils/gpu_delta/encoder.py",
    "miles/utils/gpu_delta/nvcomp.py",
    "miles/utils/gpu_delta/publication.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_path(path, root=None):
    """Require the literal absolute path, including all parents, to be canonical."""
    path = Path(path)
    _require(path.is_absolute() and ".." not in path.parts, "Expected absolute contained path")
    _require(path.resolve(strict=True) == path, "Symlink or noncanonical path is not reusable")
    if root is not None:
        _require(path != root and root in path.parents, "Path escapes saved fixture/input root")
    return path


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_file(path, expected=None, json_data=False):
    """Hash a stable regular file without following a final symlink or copying payloads."""
    path = safe_path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        before = os.fstat(source.fileno())
        _require(stat.S_ISREG(before.st_mode), "Expected a regular file")
        _require(not json_data or before.st_size <= JSON_LIMIT, "JSON metadata exceeds bound")
        digest, chunks, total = hashlib.sha256(), [], 0
        while block := source.read(8 << 20):
            total += len(block)
            digest.update(block)
            if json_data:
                chunks.append(block)
            _require(total <= before.st_size, "File grew while verifying")
        _require(
            total == before.st_size and _stamp(before) == _stamp(os.fstat(source.fileno())) == _stamp(path.stat()),
            "File changed while verifying",
        )
    actual = digest.hexdigest()
    _require(expected is None or actual == _sha(expected), "File SHA256 differs")
    ref = {"path": str(path), "sha256": actual, "nbytes": total}
    if not json_data:
        return ref
    value = json.loads(b"".join(chunks), parse_constant=lambda _: _require(False, "Nonfinite JSON"))
    return value, ref


def _option(argv, name):
    _require(argv.count(name) == 1, f"Producer SPEC must explicitly contain {name}")
    start = argv.index(name) + 1
    values = []
    while start < len(argv) and not argv[start].startswith("--"):
        values.append(argv[start])
        start += 1
    _require(values, f"Empty producer option {name}")
    return values


def _producer(spec_path, spec_sha, root, cache, args, metadata):
    spec, ref = read_file(spec_path, _sha(spec_sha), json_data=True)
    argv = spec["argv"]
    _require(isinstance(argv, list) and all(isinstance(v, str) for v in argv), "Invalid producer argv")
    expected = {
        "--output": str(root),
        "--prepared-inputs": str(cache),
        "--model": str(args.model),
        "--versions": str(args.versions),
        "--seed": str(args.seed),
        "--frame-bytes": str(args.frame_bytes),
        "--sender-gpus": str(args.sender_gpus),
    }
    for flag, value in expected.items():
        _require(_option(argv, flag) == [value], f"Producer setting differs: {flag}")
    _require(_option(argv, "--ratio") == [str(args.ratio)], "Producer mutation ratio differs")
    _require(not any(flag.startswith("--reuse-") for flag in argv), "Reference the original encoder run")
    _require("on" in _option(argv, "--hashing"), "Producer did not encode SHA-on payloads")
    for arm in args.arms:
        _require(arm["codec"] in _option(argv, "--codecs"), "Producer codec differs")
    # Producer metadata pins are explicit. Do not infer them from today's
    # receiver source or from cache file names alone.
    for relative, file_ref in metadata.items():
        _require(
            spec["cache_files"].get(f"{cache.name}/{relative}") == file_ref["sha256"],
            "Producer input-cache identity differs",
        )
    pins = {}
    for suffix in PRODUCER_FILES:
        matches = [(name, sha) for name, sha in spec["guard_files"].items() if name.endswith("/" + suffix)]
        _require(len(matches) == 1, "Producer source provenance missing/ambiguous")
        pins[matches[0][0]] = _sha(matches[0][1])
    _require(argv[0].endswith("/tests/manual/gpu_delta/bench_gpu_delta.py"), "Wrong producer entrypoint")
    return spec, {
        "spec": ref,
        "source_files": pins,
        "package_versions": spec.get("package_versions", "unknown; retain original setup evidence"),
    }


def _check_manifest(path, descriptor, owners, plan, metadata_refs, payload_refs, validate_codec):
    manifest, ref = read_file(path, descriptor["manifest_sha256"], json_data=True)
    metadata_refs.append(ref)
    validate_codec(manifest)  # Production codec/frame admission, unchanged.
    _require(all(manifest[k] == descriptor[k] for k in IDENTITY), "Manifest identity differs")
    tensors, files = [], []
    for owner, record in owners:
        shard = record["shard"]
        _require(shard["metadata"] == {k: descriptor[k] for k in IDENTITY}, "Owner publication differs")
        _require(sorted(t["name"] for t in shard["tensors"]) == sorted(owner["tensors"]), "Owner tensor set differs")
        _require([f["name"] for f in shard["files"]] == [f"owner-{owner['owner']:05d}.bin"], "Owner payload differs")
        tensors.extend(shard["tensors"])
        files.extend(shard["files"])
    _require(len({t["name"] for t in tensors}) == len(tensors), "Duplicate tensor ownership")
    _require(manifest["tensors"] == sorted(tensors, key=lambda t: t["name"]), "Sealed tensors differ from owners")
    _require(manifest["files"] == sorted(files, key=lambda f: f["name"]), "Sealed files differ from owners")
    _require(
        [{k: t[k] for k in PLAN_FIELDS} for t in manifest["tensors"]] == sorted(plan, key=lambda t: t["name"]),
        "Publication canonical plan differs",
    )
    _require(not (path.parent / "manifest.json.pending").exists(), "Publication sealing incomplete")
    for file in manifest["files"]:
        _require(type(file["nbytes"]) is int and file["nbytes"] >= 0, "Invalid payload extent")
        payload = safe_path(path.parent / file["name"], path.parent)
        _require(payload.parent == path.parent, "Nested payload path")
        proof = read_file(payload, _sha(file["sha256"]))
        _require(proof["nbytes"] == file["nbytes"], "Payload size differs")
        payload_refs.append(proof)
    return manifest


def admit(args, prepared, plan, assignments, validate_codec, accounting):
    """Verify saved publications and return small immutable references, one per arm.

    ``plan`` and ``assignments`` come from the benchmark's production merge_plans
    and existing whole-layer partitioner. No current encoder is invoked here.
    """
    root, cache = safe_path(args.reuse_fixtures), safe_path(args.prepared_inputs)
    _require(root.is_dir() and cache.is_dir(), "Saved run and prepared cache must exist")
    _require(not any(a["skip_payload_hash"] for a in args.arms), "Saved-delta mode requires SHA-on payloads")
    _require(args.sender_gpus is not None, "Saved-delta mode requires explicit --sender-gpus")
    _require(
        args.output != root and root not in args.output.parents and args.output not in root.parents,
        "New output must be disjoint from the saved run",
    )
    metadata = {}
    for relative in ("READY.json", "data/index.json", "inventory/inventory.json"):
        path = safe_path(cache / relative, cache)
        if relative == "READY.json":
            value, metadata[relative] = read_file(path, json_data=True)
            _require(value == prepared, "Prepared READY changed after admission")
        else:
            # Inventory is ~381 MB. The production loader already parsed it;
            # provenance needs its SHA, not a second in-memory JSON copy.
            metadata[relative] = read_file(path)
    for file in prepared["files"]:
        p = safe_path(cache / file["path"], cache)
        _require(p.is_file() and p.stat().st_size == file["nbytes"], "Prepared input file extent differs")
    spec, provenance = _producer(
        args.reuse_producer_spec, args.reuse_producer_spec_sha256, root, cache, args, metadata
    )
    _require(
        spec["prepared_settings_sha256"] == hashlib.sha256(_canonical(prepared["settings"])).hexdigest(),
        "Producer prepared settings differ",
    )
    arguments, arguments_ref = read_file(root / "arguments.json", json_data=True)
    _require(arguments["output"] == str(root), "Saved producer output differs")
    for key in ("model", "prepared_inputs", "versions", "seed", "ratio", "frame_bytes"):
        current = getattr(args, key)
        _require(arguments[key] == (str(current) if isinstance(current, Path) else current), "Saved arguments differ")
    # Older producers used None for all visible GPUs. The pinned producer
    # record resolves that default; exact owner-plan equality below still
    # checks the actual partition, not merely the declared device count.
    _require(arguments["sender_gpus"] in (None, args.sender_gpus), "Saved sender count differs")
    workflow, workflow_ref = read_file(root / "workflow.json", json_data=True)
    _require(not any(p["status"] == "RUNNING" for p in workflow["phases"]), "Saved producer workflow is still active")
    inputs, inputs_ref = read_file(root / "inputs.json", json_data=True)
    _require(
        inputs["path"] == str(cache) and inputs["id"] == args.input_cache_id and inputs["preparation"] == prepared,
        "Saved input reference differs",
    )
    result = {}
    for arm in args.arms:
        phase = [p for p in workflow["phases"] if p["phase"] == "fixture" and p["arm"] == arm["name"]]
        _require(
            len(phase) == 1 and phase[0]["status"] == "PASS" and phase[0]["exit_code"] == 0,
            "Saved encoder phase is not complete",
        )
        _require(arm in workflow["arms"] and arm in arguments["arms"], "Saved arm differs")
        directory = safe_path(root / "arms" / arm["name"] / "fixture", root)
        fixture, fixture_ref = read_file(directory / "fixture.json", json_data=True)
        owner_plan, owner_ref = read_file(directory / "owner-plan.json", json_data=True)
        _require(owner_plan == assignments == fixture["sender_owners"], "Sender owner partition differs")
        _require(
            fixture["sender_gpus"] == len(assignments) and fixture["owner_plan"] == "owner-plan.json",
            "Sender owner count/path differs",
        )
        expected = {
            "plan_digest": prepared["plan_digest"],
            "rate": prepared["rate"],
            "requested_zstd_ratio": args.ratio,
            "seed": args.seed,
            "input_cache_id": args.input_cache_id,
            "codec": arm["codec"],
            "arm": arm["name"],
            "target_checkpoint": str(safe_path(cache / "target-checkpoint", cache)),
        }
        _require(all(fixture[k] == value for k, value in expected.items()), "Fixture settings/target differ")
        stream = hashlib.sha256(
            _canonical(
                {"seed": args.seed, "plan": prepared["plan_digest"], "rate": prepared["rate"], "arm": arm["name"]}
            )
        ).hexdigest()
        _require(
            fixture["stream_id"] == stream and len(fixture["rounds"]) == args.versions, "Incomplete fixture chain"
        )
        refs, payloads = [fixture_ref, owner_ref, arguments_ref, workflow_ref, inputs_ref, *metadata.values()], []
        for version, row in enumerate(fixture["rounds"], 1):
            identity = {
                "stream_id": stream,
                "publication_id": f"{stream}:{version}",
                "plan_digest": prepared["plan_digest"],
                "base_version": version - 1,
                "target_version": version,
                "codec": arm["codec"],
                "frame_bytes": args.frame_bytes,
                "payload_checksum_format": "sha256",
            }
            _require(
                row["version"] == version and set(row["publications"]) == {arm["codec"]}, "Fixture version differs"
            )
            descriptor = row["publications"][arm["codec"]]
            path = safe_path(directory / arm["codec"] / f"v{version}" / "manifest.json", directory)
            _require(
                descriptor["manifest_path"] == str(path) and all(descriptor[k] == v for k, v in identity.items()),
                "Publication identity/path differs",
            )
            owners = []
            for owner in assignments:
                prefix = f"owner-{owner['owner']:05d}"
                _require(not (directory / f"{prefix}-failed.json").exists(), "Encoder owner failed")
                record, ref = read_file(directory / f"{prefix}-v{version}.json", json_data=True)
                refs.append(ref)
                _require(
                    record["version"] == version
                    and all(record["metrics"][k] == owner[k] for k in ("owner", "gpu", "groups", "canonical_bytes")),
                    "Owner record differs",
                )
                _require(record["metrics"]["payload_checksum_format"] == "sha256", "Owner hash policy differs")
                owners.append((owner, record))
            _check_manifest(path, descriptor, owners, plan, refs, payloads, validate_codec)
            _require(
                row["canonical_bytes"] == prepared["canonical_bytes"]
                and row["changed_bytes"] == sum(r["changed_bytes"] for _, r in owners),
                "Fixture byte accounting differs",
            )
            _require(
                row["compression"]["owners"] == [r["metrics"] for _, r in owners], "Historical sender metrics differ"
            )
            _require(row["accounting"][arm["codec"]] == accounting(descriptor), "Publication accounting differs")
        # Seal a reference only after every round/owner/payload has passed. It
        # does not mutate or create a completion marker inside the old run.
        result[arm["name"]] = {
            "schema": "gpu_delta_saved_fixture_v1",
            "mode": "saved_delta_apply",
            "fixture": fixture_ref,
            "producer": provenance,
            "metadata": refs,
            "payloads": payloads,
            "historical_fixture_phase": phase[0],
            "sender_measurement_scope": "historical_saved_fixture",
            "payload_bytes_verified": True,
            "prepared_checkpoint_bytes_verified": False,
            "limits": [
                "Prepared checkpoint identity uses original metadata/settings/file sizes, not full byte hashes.",
                "Fresh receiver admission remains mandatory; target equivalence requires the opt-in oracle.",
                "No current sender/compression performance or changed-encoder validation.",
            ],
        }
    return result


def resolve(arm_output):
    """Resolve explicit metadata; never silently choose an external directory."""
    reference = arm_output / "fixture-reuse.json"
    if not reference.exists():
        return arm_output / "fixture", None
    value, _ = read_file(reference, json_data=True)
    _require(
        value["schema"] == "gpu_delta_saved_fixture_v1" and value["mode"] == "saved_delta_apply",
        "Unsupported fixture reference",
    )
    fixture, _ = read_file(value["fixture"]["path"], value["fixture"]["sha256"], json_data=True)
    _require(fixture["arm"] == arm_output.name, "Referenced arm differs")
    return Path(value["fixture"]["path"]).parent, value


def record_producer(args, prepared):
    """Record fresh encoder provenance; workflow completion is admitted separately."""
    source_root = Path(__file__).resolve().parents[3]
    argv = [str(source_root / "tests/manual/gpu_delta/bench_gpu_delta.py")]
    for name in ("model", "output", "prepared_inputs", "versions", "seed", "ratio", "frame_bytes", "sender_gpus"):
        argv.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    argv.extend(["--codecs", *args.codecs, "--hashing", *args.hashing])
    value = {
        "schema": "gpu_delta_benchmark_producer_v1",
        "argv": argv,
        "cache_files": {
            f"{args.prepared_inputs.name}/{name}": read_file(args.prepared_inputs / name)["sha256"]
            for name in ("READY.json", "data/index.json", "inventory/inventory.json")
        },
        "prepared_settings_sha256": hashlib.sha256(_canonical(prepared["settings"])).hexdigest(),
        "guard_files": {str(source_root / name): read_file(source_root / name)["sha256"] for name in PRODUCER_FILES},
        "package_versions": "unknown; retain the original environment/setup evidence",
    }
    with (args.output / "producer-spec.json").open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
