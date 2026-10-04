"""Admission and ownership checks for the sole-codec GPU delta benchmarks."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

_MODULE = Path(__file__).parents[2] / "manual" / "gpu_delta" / "bench_gpu_delta.py"
_spec = importlib.util.spec_from_file_location("bench_gpu_delta", _MODULE)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _descriptions(engine_count, tensors):
    size = 8 // engine_count
    descriptions = []
    for engine in range(engine_count):
        participants = []
        for rank in range(size):
            views = []
            for tensor in tensors:
                shape = tensor["shape"]
                slices = [[0, n] for n in shape]
                if len(shape) == 2:
                    slices[0] = [rank * shape[0] // size, (rank + 1) * shape[0] // size]
                views.append(tensor | {"views": [{"id": f"view-{size}-{rank}", "slices": slices}]})
            identity = {
                "engine_id": f"engine-{engine:05d}",
                "rank_id": f"{engine}-{rank}",
                "pid": 100 + engine * size + rank,
                "start_ticks": 42,
                "dp_rank": rank,
                "tp_rank": rank,
                "pp_rank": 0,
                "host_cache_id": f"engine-host-{engine}",
            }
            participants.append({"identity": identity, "plan": {"codec": "snappy-zstd", "tensors": views}})
        descriptions.append({"success": True, "participants": participants})
    return descriptions


@pytest.mark.parametrize("engine_count", [1, 2])
def test_topology_and_generation_cover_each_original_engine_rank(monkeypatch, engine_count):
    import asyncio

    ports = list(range(31100, 31100 + engine_count))
    specs = bench._engine_specs(ports)
    assert [spec["parallel_size"] for spec in specs] == [8 // engine_count] * engine_count
    assert [gpu for spec in specs for gpu in spec["gpu_ids"].split(",")] == list(map(str, range(8)))
    cohort = bench.negotiate_cohort(_descriptions(engine_count, []))
    bench._validate_cohort(cohort, engine_count)
    calls = []

    async def request(client, endpoint, payload):
        await asyncio.sleep(0)
        calls.append((client, payload["routed_dp_rank"]))
        assert endpoint == "generate" and payload["sampling_params"] == {"temperature": 0, "max_new_tokens": 32}
        return {"client": client, "rank": payload["routed_dp_rank"]}

    monkeypatch.setattr(bench, "_request", request)
    records = asyncio.run(bench._generation(list(range(engine_count))))
    assert set(calls) == {(engine, rank) for engine in range(engine_count) for rank in range(8 // engine_count)}
    assert sum(len(record["engines"]) for record in records) == 8
    assert all(record["engine_ids"] == list(cohort.engine_ids) for record in records)
    devices = "".join(f"{i}, GPU-{i}\n" for i in range(8))
    processes = "".join(f"{100+i}, GPU-{i}\n" for i in range(8))
    monkeypatch.setattr(
        bench.subprocess,
        "check_output",
        lambda command, **kw: devices if command[1].startswith("--query-gpu=") else processes,
    )
    monkeypatch.setattr(bench, "_pid_candidates", lambda pid: [pid])
    assert len(bench._capture_gpu_processes(cohort, ports)["participants"]) == 8
    processes = processes.replace("100, GPU-0", "100, GPU-7")
    assert bench._capture_gpu_processes(cohort, ports)["status"] == "UNQUALIFIED_PID_NAMESPACE"


def _tiny_fixture(tmp_path):

    import numpy as np
    import zstandard
    from safetensors.numpy import save_file

    tensors = [
        {"name": "matrix.weight", "dtype": "U8", "shape": [8, 2], "encoding": "xor_bytes"},
        {"name": "norm.weight", "dtype": "F32", "shape": [2], "encoding": "raw_bytes"},
    ]
    old_plan, _, old_digest = bench.merge_plans(_descriptions(1, tensors))
    fresh = _descriptions(2, tensors)
    _, _, new_digest = bench.merge_plans(fresh)
    inventory = tmp_path / "inventory.json"
    bench._save(inventory, {"descriptions": fresh, "plan_digest": new_digest})
    model, target, source = (tmp_path / name for name in ("model", "target", "source"))
    for directory in (model, target, source):
        directory.mkdir()
    before = np.arange(16, dtype=np.uint8).reshape(8, 2)
    mask = np.zeros(16, dtype=np.uint8)
    mask[[1, 4, 14]] = 1
    after = before ^ mask.reshape(8, 2)
    raw = np.array([1.0, 2.0], dtype=np.float32)
    changed = np.array([1.5, 2.5], dtype=np.float32)
    save_file({"matrix.weight": before, "norm.weight": raw}, model / "model.safetensors")
    save_file({"matrix.weight": after, "norm.weight": changed}, target / "model.safetensors")
    writer = bench.PublicationWriter(
        source / "v1", stream_id="original", base_version=0, target_version=1, plan_digest=old_digest
    )
    # Fixed raw Snappy fixture: length 16 followed by one 16-byte literal.
    inner = b"\x10\x3c" + mask.tobytes()
    payload = zstandard.ZstdCompressor(level=1).compress(inner)
    frames = [{"decoded_offset": 0, "decoded_bytes": 16, "encoded_offset": 0, "encoded_bytes": len(inner)}]
    outer = {
        "decoded_bytes": len(inner),
        "encoded_bytes": len(payload),
        "frames": [
            {"decoded_offset": 0, "decoded_bytes": len(inner), "encoded_offset": 0, "encoded_bytes": len(payload)}
        ],
    }
    writer.add_gpu_outer_tensor(
        "matrix.weight", frames, payload, outer, changed_bytes=3, dtype="U8", shape=[8, 2], views=old_plan[0]["views"]
    )
    writer.add_raw_tensor("norm.weight", raw, changed, dtype="F32", shape=[2], views=old_plan[1]["views"])
    descriptor = writer.finish()
    fixture = {
        "codec": "snappy-zstd",
        "plan_digest": old_digest,
        "stream_id": "original",
        "target_checkpoint": str(target),
        "rounds": [{"version": 1, "canonical_bytes": 24, "publications": {"snappy-zstd": descriptor}}],
    }
    bench._save(source / "fixture.json", fixture)
    output = tmp_path / "rebound"
    output.mkdir()
    return Namespace(model=model, fixture=source, inventory=inventory, output=output), before, after, changed


def test_rebind_preserves_payload_and_exact_canonical_target(tmp_path):
    import json

    import numpy as np
    import zstandard

    args, before, after, changed = _tiny_fixture(tmp_path)
    source_fixture = (args.fixture / "fixture.json").read_bytes()
    bench._rebind(args)
    fixture = json.loads((args.output / "fixture.json").read_text())
    proof = json.loads((args.output / "rebind.json").read_text())
    old = json.loads(source_fixture)
    assert proof["status"] == "PASS" and fixture["target_checkpoint"] == old["target_checkpoint"]
    assert fixture["stream_id"] != old["stream_id"] and fixture["plan_digest"] != old["plan_digest"]
    assert (args.fixture / "fixture.json").read_bytes() == source_fixture
    for file in proof["rounds"][0]["files"]:
        assert Path(file["source"]).stat().st_ino == Path(file["target"]).stat().st_ino
        assert bench.sha256(Path(file["target"]).read_bytes()) == file["sha256"]
    publication = fixture["rounds"][0]["publications"]["snappy-zstd"]
    manifest = json.loads(Path(publication["manifest_path"]).read_text())
    for tensor in manifest["tensors"]:
        location = tensor.get("outer", tensor.get("raw"))
        payload = (Path(publication["manifest_path"]).parent / location["file"]).read_bytes()
        encoded = payload[location["encoded_offset"] : location["encoded_offset"] + location["encoded_bytes"]]
        if tensor["encoding"] == "xor_bytes":
            inner = zstandard.ZstdDecompressor().decompress(encoded)
            assert inner[:2] == b"\x10\x3c" and len(inner) == 18
            mask = np.frombuffer(inner[2:], dtype=np.uint8)
            actual, expected = before.reshape(-1) ^ mask, after.reshape(-1)
        else:
            actual, expected = np.frombuffer(encoded, dtype=np.uint8), changed.view(np.uint8)
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("corruption", ["canonical", "payload", "manifest", "host"])
def test_rebind_rejects_incompatible_or_changed_inputs(tmp_path, corruption):
    import json

    args, _, _, _ = _tiny_fixture(tmp_path)
    if corruption == "canonical":
        inventory = json.loads(args.inventory.read_text())
        for engine in inventory["descriptions"]:
            for participant in engine["participants"]:
                participant["plan"]["tensors"][0]["encoding"] = "raw_bytes"
        inventory["plan_digest"] = bench.merge_plans(inventory["descriptions"])[2]
        bench._save(args.inventory, inventory)
    elif corruption == "host":
        inventory = json.loads(args.inventory.read_text())
        inventory["descriptions"][1]["participants"][0]["identity"]["host_cache_id"] = "other-container"
        bench._save(args.inventory, inventory)
    else:
        path = args.fixture / "v1" / ("owner-00000.bin" if corruption == "payload" else "manifest.json")
        path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError):
        bench._rebind(args)
    assert not (args.output / "fixture.json").exists()
