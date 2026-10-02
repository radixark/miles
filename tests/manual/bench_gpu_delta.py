"""One EP8 GLM5.2 W4A16 engine: direct GPU delta fixture and timing harness.

Requires paired SGLang GPU-delta sources, eight Blackwell GPUs, the Miles image,
prebuilt nvCOMP >=5.3 and an immutable NVFP4 checkpoint with bundled MTP weights.
No private model names/paths or cluster launch commands are embedded here.

Example (run from the Miles checkout with paired SGLang on PYTHONPATH)::

    python tests/manual/bench_gpu_delta.py inventory --model /models/base --output /data/inventory
    python tests/manual/bench_gpu_delta.py fixture --model /models/base \
        --inventory /data/inventory/inventory.json --output /data/fixture
    WEIGHT_DELTA_CODEC=zstd WEIGHT_DELTA_TIMING=1 \
        python tests/manual/bench_gpu_delta.py run --model /models/base \
        --fixture /data/fixture --output /data/zstd

Repeat run with Snappy. Each run starts one fresh engine;
never reset a live delta stream with a disk reload. ``oracle`` starts from the
altered target checkpoint while keeping the original static draft, outside update
timing. Compare its generation/logprob records with the last delta round.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
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
from contextlib import ExitStack
from pathlib import Path

import httpx
import numpy as np
import zstandard

from miles.backends.training_utils.weight_update.gpu_delta_session import activate_publication, merge_plans
from miles.utils.gpu_delta_publication import (
    DTYPE_BYTES,
    FRAME_BYTES,
    PublicationWriter,
    canonical_json,
    settings_from_env,
    sha256,
    snappy_zstd_from_env,
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


def _mutate(raw, *, name, dtype, seed, version, rate):
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


def _fixture(args):
    inventory = json.loads(args.inventory.read_text())
    plan, _, digest = merge_plans(inventory["descriptions"])
    index = _tensor_index(args.model)
    for spec in plan:
        actual = index[spec["name"]]
        if spec["shape"] != actual["shape"] or spec["dtype"] != actual["dtype"]:
            raise ValueError(f"Inventory/checkpoint mismatch: {spec['name']}")
    target = args.output / "altered-checkpoint"
    target.mkdir()
    # A real independent copy: no hardlinks that could mutate the source model.
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
        "calibration": "Zstd level-1 sample-frame estimate; one shared mutation rate for all codecs/versions. Final ratios include separately reported metadata and padding.",
        "canonical_denominator": "Sum of mutable canonical tensors in the negotiated receiver plan; excludes frozen draft and non-updated checkpoint entries.",
        "changed_bytes_definition": "Count of unequal storage bytes, not changed bits or compressed size.",
        "codecs": args.codecs,
        "inner_snappy_origin": "CPU snappy.compress in the fixture builder; independent frames compatible with nvCOMP decode. No GPU producer ran while building this fixture.",
    }
    denominator = sum(index[t["name"]]["nbytes"] for t in plan)
    for version in range(1, args.versions + 1):
        writers = {
            codec: PublicationWriter(
                args.output / codec / f"v{version}",
                stream_id=stream_id,
                base_version=version - 1,
                target_version=version,
                plan_digest=digest,
                codec=codec,
                publication_id=f"{stream_id}:{version}",
            )
            for codec in args.codecs
        }
        started = time.monotonic()
        changed = 0
        try:
            for tensor in plan:
                spec = index[tensor["name"]]
                with (target / spec["shard"]).open("r+b") as file:
                    file.seek(spec["offset"])
                    before = np.frombuffer(file.read(spec["nbytes"]), dtype=np.uint8)
                    after = _mutate(
                        before, name=tensor["name"], dtype=spec["dtype"], seed=args.seed, version=version, rate=rate
                    )
                    changed += int(np.count_nonzero(before != after))
                    for writer in writers.values():
                        writer.add_tensor(
                            tensor["name"],
                            before,
                            after,
                            dtype=spec["dtype"],
                            shape=spec["shape"],
                            views=tensor["views"],
                            encoding=tensor["encoding"],
                        )
                    file.seek(spec["offset"])
                    file.write(after)
            publications = {codec: writer.finish() for codec, writer in writers.items()}
        finally:
            for writer in writers.values():
                writer.close()
        accounting = {}
        for codec, publication in publications.items():
            path = Path(publication["manifest_path"])
            manifest = json.loads(path.read_text())
            payload = sum(item["nbytes"] for item in manifest["files"])
            encoded = sum(frame["encoded_bytes"] for tensor in manifest["tensors"] for frame in tensor["frames"])
            accounting[codec] = {
                "encoded_frame_bytes": encoded,
                "payload_file_bytes": payload,
                "alignment_bytes": payload - encoded,
                "manifest_bytes": path.stat().st_size,
                "publication_bytes": payload + path.stat().st_size,
            }
        row = {
            "version": version,
            "publications": publications,
            "canonical_bytes": denominator,
            "changed_bytes": changed,
            "changed_byte_fraction": changed / denominator,
            "accounting": accounting,
            "ratios": {codec: size["publication_bytes"] / denominator for codec, size in accounting.items()},
            "encoded_frame_ratios": {
                codec: size["encoded_frame_bytes"] / denominator for codec, size in accounting.items()
            },
            "encode_and_target_write_s": time.monotonic() - started,
        }
        report["rounds"].append(row)
        _save(args.output / "fixture.json", report)
        print(json.dumps({k: v for k, v in row.items() if k != "publications"}), flush=True)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_payloads(stack, manifest_path, manifest):
    files = {}
    for item in manifest["files"]:
        if Path(item["name"]).name != item["name"] or item["name"] in files:
            raise ValueError("Expected distinct local publication payload files")
        path = manifest_path.parent / item["name"]
        if path.stat().st_size != item["nbytes"] or _file_sha256(path) != item["sha256"]:
            raise ValueError(f"Source encoded file changed: {path}")
        files[item["name"]] = (stack.enter_context(path.open("rb")), item["nbytes"])
    return files


def _payload_bytes(files, frame):
    source, nbytes = files[frame["file"]]
    start, size = frame["encoded_offset"], frame["encoded_bytes"]
    if type(start) is not int or type(size) is not int or start < 0 or size <= 0 or start + size > nbytes:
        raise ValueError("Invalid encoded source frame range")
    source.seek(start)
    payload = source.read(size)
    if len(payload) != size:
        raise ValueError("Truncated encoded source frame")
    return payload


def _verify_wrapped_publication(publication, source_manifest, source_files):
    path = Path(publication["manifest_path"])
    raw = path.read_bytes()
    if sha256(raw) != publication["manifest_sha256"]:
        raise ValueError("Derived manifest digest differs")
    manifest = json.loads(raw)
    if manifest["protocol_version"] != 3 or manifest["codec_profile"] != source_manifest["codec_profile"].removesuffix("-v1") + "-zstd-v1":
        raise ValueError("Derived publication profile differs")
    for key in ("plan_digest", "stream_id", "publication_id", "base_version", "target_version"):
        if manifest[key] != source_manifest[key]:
            raise ValueError(f"Derived publication identity differs: {key}")
    original = {tensor["name"]: tensor for tensor in source_manifest["tensors"]}
    if len(original) != len(manifest["tensors"]):
        raise ValueError("Derived tensor inventory differs")
    inner_bytes = outer_bytes = decoded_bytes = 0
    with ExitStack() as stack:
        files = _open_payloads(stack, path, manifest)
        for tensor in manifest["tensors"]:
            previous = original.pop(tensor["name"])
            for key in ("name", "dtype", "shape", "nbytes", "byte_order", "encoding", "views", "changed_bytes"):
                if tensor[key] != previous[key]:
                    raise ValueError(f"Derived canonical definition changed: {tensor['name']} {key}")
            if not previous["frames"]:
                if tensor["frames"] or "outer" in tensor:
                    raise ValueError("Empty XOR tensor acquired a payload")
                continue
            outer = tensor["outer"]
            if outer["codec"] != "zstd":
                raise ValueError("Expected CPU Zstd envelope")
            arena = zstandard.ZstdDecompressor().decompress(
                _payload_bytes(files, outer), max_output_size=outer["decoded_bytes"]
            )
            if len(arena) != outer["decoded_bytes"]:
                raise ValueError("Derived inner arena length differs")
            end = 0
            for old, new in zip(previous["frames"], tensor["frames"], strict=True):
                for key in ("decoded_offset", "decoded_bytes", "encoded_bytes", "codec", "encoded_sha256"):
                    if old[key] != new[key]:
                        raise ValueError("Derived inner frame metadata changed")
                start, size = new["encoded_offset"], new["encoded_bytes"]
                if start != (end + 15) // 16 * 16 or any(arena[end:start]):
                    raise ValueError("Derived inner arena padding differs")
                expected = _payload_bytes(source_files, old)
                if arena[start : start + size] != expected or sha256(expected) != old["encoded_sha256"]:
                    raise ValueError("Derived inner Snappy/raw bytes differ")
                end = start + size
                inner_bytes += size
            if end != len(arena):
                raise ValueError("Derived inner arena has trailing bytes")
            outer_bytes += outer["encoded_bytes"]
            decoded_bytes += len(arena)
    if original:
        raise ValueError("Derived tensor inventory is incomplete")
    payload_bytes = sum(item["nbytes"] for item in manifest["files"])
    return {
        "encoded_frame_bytes": inner_bytes,
        "outer_encoded_bytes": outer_bytes,
        "outer_decoded_arena_bytes": decoded_bytes,
        "payload_file_bytes": payload_bytes,
        "alignment_bytes": payload_bytes - outer_bytes,
        "manifest_bytes": len(raw),
        "publication_bytes": payload_bytes + len(raw),
        "exact_inner_payloads_verified": True,
    }


def _wrap_publication(publication, row, fixture, output):
    path = Path(publication["manifest_path"])
    raw = path.read_bytes()
    if sha256(raw) != publication["manifest_sha256"]:
        raise ValueError("Source fixture manifest digest differs")
    manifest = json.loads(raw)
    frame_bytes = {"snappy-independent-64kib-v1": 1 << 16, "snappy-independent-1mib-v1": FRAME_BYTES}.get(manifest["codec_profile"])
    if manifest["protocol_version"] != 2 or frame_bytes is None:
        raise ValueError("wrap-fixture requires unwrapped receiver-compatible Snappy publications")
    expected = {
        "plan_digest": fixture["plan_digest"], "stream_id": fixture["stream_id"],
        "base_version": row["version"] - 1, "target_version": row["version"],
    }
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("Source fixture plan/stream/version differs")
    with ExitStack() as stack:
        files = _open_payloads(stack, path, manifest)
        writer = PublicationWriter(
            output / "snappy-zstd" / f"v{row['version']}", codec="snappy", snappy_zstd=True,
            frame_bytes=frame_bytes, publication_id=manifest["publication_id"], **expected,
        )
        try:
            for tensor in manifest["tensors"]:
                payloads = [_payload_bytes(files, frame) for frame in tensor["frames"]]
                writer.add_encoded_tensor(
                    tensor["name"], tensor["frames"], payloads, changed_bytes=tensor["changed_bytes"],
                    dtype=tensor["dtype"], shape=tensor["shape"], views=tensor["views"], encoding=tensor["encoding"],
                )
            derived = writer.finish()
        finally:
            writer.close()
        accounting = _verify_wrapped_publication(derived, manifest, files)
    return derived, accounting


def _wrap_fixture(args):
    source_path = args.fixture / "fixture.json"
    raw = source_path.read_bytes()
    if sha256(raw) != args.fixture_sha256:
        raise ValueError("Source fixture digest differs from --fixture-sha256")
    source = json.loads(raw)
    report = copy.deepcopy(source)
    if "snappy-zstd" in report["codecs"] or any("snappy" not in row["publications"] for row in report["rounds"]):
        raise ValueError("Expected an unwrapped fixture with Snappy in every round")
    receipt = {
        "status": "deriving", "source_fixture": str(source_path.resolve()), "source_fixture_sha256": sha256(raw),
        "plan_digest": source["plan_digest"], "stream_id": source["stream_id"], "target_checkpoint": source["target_checkpoint"],
        "proof": "Per-tensor CPU Zstd envelopes preserve every inner Snappy/raw byte, canonical definition and version. No model tensor bytes are read or re-exported.",
        "inner_snappy_origin": source.get("inner_snappy_origin", "Historical fixture builder CPU snappy.compress; nvCOMP-compatible independent frames. This derivation does not run GPU compression."),
        "rounds": [],
    }
    _save(args.output / "derivation.json", receipt)
    try:
        for version, row in enumerate(report["rounds"], start=1):
            if row["version"] != version:
                raise ValueError("Expected consecutive fixture versions beginning at 1")
            started = time.monotonic()
            publication, accounting = _wrap_publication(row["publications"]["snappy"], row, source, args.output)
            row["publications"]["snappy-zstd"] = publication
            row["accounting"]["snappy-zstd"] = accounting
            row["ratios"]["snappy-zstd"] = accounting["publication_bytes"] / row["canonical_bytes"]
            row["encoded_frame_ratios"]["snappy-zstd"] = accounting["encoded_frame_bytes"] / row["canonical_bytes"]
            receipt["rounds"].append({"version": version, "publication": publication, "accounting": accounting, "derive_and_verify_s": time.monotonic() - started})
            _save(args.output / "derivation.json", receipt)
        report["codecs"].append("snappy-zstd")
        report["outer_zstd_derivation"] = {key: value for key, value in receipt.items() if key not in {"status", "rounds"}}
        _save(args.output / "fixture.json", report)
        receipt.update(status="completed", fixture_sha256=_file_sha256(args.output / "fixture.json"))
        _save(args.output / "derivation.json", receipt)
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save(args.output / "derivation.json", receipt)
        raise


async def _request(client, endpoint, payload=None, *, timeout=1200):
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
async def _engines(args, model):
    # Keep CPU-only fixture creation independent of the serving client imports.
    from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient

    processes, logs, clients = [], [], []
    commands = []
    try:
        for engine, port in enumerate(args.ports):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            config = SERVER_ARGS | {
                "model_path": str(model),
                "speculative_draft_model_path": str(args.model),
                "host": "127.0.0.1",
                "port": port,
                "random_seed": 1238,
            }
            config_path = args.output / f"engine-{engine}.json"
            _save(config_path, config)
            code = (
                "import json,sys; from sglang.srt.server_args import ServerArgs; "
                "from sglang.srt.entrypoints.http_server import launch_server; "
                "launch_server(ServerArgs(**json.load(open(sys.argv[1]))))"
            )
            command = [sys.executable, "-c", code, str(config_path)]
            env = os.environ | SERVER_ENV | {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
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
            commands.append({"command": command, "pid": process.pid, "gpu_ids": env["CUDA_VISIBLE_DEVICES"]})
        _save(
            args.output / "launch.json",
            {
                "engines": commands,
                "feature_env": {key: os.environ.get(key) for key in ("WEIGHT_DELTA_CODEC", "WEIGHT_DELTA_SNAPPY_ZSTD", "WEIGHT_DELTA_TIMING")},
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
    for dp_rank in range(8):
        payload = {
            "text": "Explain why water freezes at low temperatures in one sentence.",
            "routed_dp_rank": dp_rank,
            "return_logprob": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        }
        values = await asyncio.gather(*[_request(client, "generate", payload) for client in clients])
        records.append({"dp_rank": dp_rank, "engines": values})
    return records


async def _run(args):
    codec, encoder = settings_from_env()
    wrapped = snappy_zstd_from_env(codec, encoder)
    fixture_key = "snappy-zstd" if wrapped else codec
    fixture = json.loads((args.fixture / "fixture.json").read_text()) if args.fixture else None
    if args.phase == "run" and any(fixture_key not in row["publications"] for row in fixture["rounds"]):
        raise ValueError(f"Fixture does not contain {fixture_key}; create or derive the required publication first")
    model = Path(fixture["target_checkpoint"]) if args.phase == "oracle" else args.model
    async with _engines(args, model) as clients:
        if args.phase == "oracle":
            _save(args.output / "target-generation.json", await _generation(clients))
            return
        descriptions = await asyncio.gather(
            *[c.get_weights_delta_info(engine_id=f"engine-{i:05d}") for i, c in enumerate(clients)]
        )
        plan, cohort, digest = merge_plans(descriptions)
        if len(cohort) != 8 or len({p["rank_id"] for p in cohort}) != 8:
            raise ValueError("Expected eight distinct native participants in the EP8 engine")
        _save(args.output / "inventory.json", {"descriptions": descriptions, "plan_digest": digest})
        if args.phase == "inventory":
            return
        if digest != fixture["plan_digest"]:
            raise ValueError("Fixture plan no longer matches current loaded models")
        _save(args.output / "base-generation.json", await _generation(clients))
        for version in fixture["rounds"]:
            started = time.monotonic()
            try:
                receipt = await activate_publication(clients, descriptions, version["publications"][fixture_key])
                result = {
                    "version": version["version"],
                    "coordinator_s": time.monotonic() - started,
                    "codec": codec,
                    "snappy_zstd": wrapped,
                    "fixture_publication": fixture_key,
                    "inner_snappy_origin": fixture.get("inner_snappy_origin", fixture.get("outer_zstd_derivation", {}).get("inner_snappy_origin")),
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
    parser.add_argument("phase", choices=("inventory", "fixture", "wrap-fixture", "run", "oracle"))
    parser.add_argument("--model", type=Path, help="Required except for CPU-only wrap-fixture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-sha256", help="Required by wrap-fixture: pinned SHA-256 of source fixture.json")
    parser.add_argument("--versions", type=int, default=3)
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=("zstd", "snappy"),
        default=["zstd", "snappy"],
        help="Fixture codecs; incompressible frames are stored raw within either profile.",
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--ratio", type=float, default=0.002)
    parser.add_argument("--ports", type=int, nargs=1, default=(31000,))
    parser.add_argument("--startup-timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.phase != "wrap-fixture" and args.model is None:
        parser.error("--model is required except for wrap-fixture")
    if args.model is not None:
        args.model = args.model.resolve(strict=True)
    if args.phase == "fixture" and args.inventory is None:
        parser.error("fixture requires --inventory")
    if args.phase in {"run", "oracle", "wrap-fixture"} and args.fixture is None:
        parser.error("run/oracle/wrap-fixture requires --fixture")
    if args.phase == "wrap-fixture" and (
        args.fixture_sha256 is None or len(args.fixture_sha256) != 64
        or any(c not in "0123456789abcdef" for c in args.fixture_sha256)
    ):
        parser.error("wrap-fixture requires a lowercase --fixture-sha256")
    if len(set(args.ports)) != 1 or any(not 0 < port < 65536 for port in args.ports):
        parser.error("--ports requires one valid TCP port")
    if args.versions < 1 or not 0 < args.ratio < 0.1:
        parser.error("versions must be positive and ratio in (0, 0.1)")
    if len(set(args.codecs)) != len(args.codecs):
        parser.error("--codecs must not contain duplicates")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.phase == "fixture":
        _fixture(args)
    elif args.phase == "wrap-fixture":
        _wrap_fixture(args)
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
