"""One EP8 GLM5.2 W4A16 engine: direct GPU delta fixture and timing harness.

Requires paired SGLang GPU-delta sources, eight Blackwell GPUs, the Miles image,
prebuilt nvCOMP >=5.3 and an immutable NVFP4 checkpoint with bundled MTP weights.
No private model names/paths or cluster launch commands are embedded here.

Example (run from the Miles checkout with paired SGLang on PYTHONPATH)::

    python tests/manual/bench_gpu_delta.py inventory --model /models/base --output /data/inventory
    python tests/manual/bench_gpu_delta.py fixture --model /models/base \
        --inventory /data/inventory/inventory.json --output /data/fixture
    WEIGHT_DELTA_CODEC=snappy-zstd WEIGHT_DELTA_TIMING=1 \
        python tests/manual/bench_gpu_delta.py run --model /models/base \
        --fixture /data/fixture --output /data/snappy-zstd

Canonical rank-zero/rank-one tensors use direct uncompressed target values when
changed; all other tensors retain compressed XOR frames. Fixture accounting keeps
direct-value traffic separate. Each run starts one fresh engine;
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

from miles.backends.training_utils.weight_update.gpu_delta_session import activate_publication, merge_plans
from miles.utils.gpu_delta_publication import (
    DTYPE_BYTES,
    FRAME_BYTES,
    PublicationWriter,
    canonical_json,
    configured_codec,
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


def _validate_fixture_codec(fixture):
    codec = configured_codec()
    if fixture.get("codec") != codec:
        raise ValueError("Fixture requires the snappy-zstd codec")
    for row in fixture["rounds"]:
        publication = row["publications"].get(codec)
        if publication is None or publication.get("protocol_version") != 4 or publication.get("codec") != codec or publication.get("frame_bytes") != FRAME_BYTES:
            raise ValueError("Fixture requires protocol 4 / snappy-zstd / 1 MiB frames")
    return codec


def _alter_fixture_tensor(target, index, tensor, *, version, seed, rate):
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
        "outer_encoded_bytes": outer_bytes,
        "outer_decoded_arena_bytes": sum(tensor.get("outer", {}).get("decoded_bytes", 0) for tensor in manifest["tensors"]),
        "outer_frames": sum(len(tensor.get("outer", {}).get("frames", [])) for tensor in manifest["tensors"]),
        "encoded_frame_bytes": inner_bytes,
        "payload_file_bytes": payload_bytes,
        "alignment_bytes": payload_bytes - raw_bytes - outer_bytes,
        "manifest_bytes": path.stat().st_size,
        "publication_bytes": payload_bytes + path.stat().st_size,
    }
    return sizes


def _fixture(args):
    import torch

    from miles.utils.gpu_delta_encoder import GpuBatchEncoder

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
        "calibration": "Plain CPU Zstd level-1 sample-frame estimate; final Snappy-Zstd size is measured separately, not forced to the requested ratio.",
        "canonical_denominator": "Mutable canonical tensors in the receiver plan; excludes frozen draft and non-updated checkpoint entries.",
        "changed_bytes_definition": "Unequal storage bytes, not changed bits or compressed size.",
        "codec": codec,
        "inner_snappy_origin": "Production GpuBatchEncoder: pinned snapshots, GPU XOR/Snappy, then one GPU Zstd submission per version. Fixture setup is excluded from receiver timing.",
    }
    encoder = GpuBatchEncoder(torch.device("cuda", torch.cuda.current_device()))
    denominator = sum(index[t["name"]]["nbytes"] for t in plan)
    raw_plan = [t for t in plan if t["encoding"] == "raw_bytes"]
    matrix_plan = [t for t in plan if t["encoding"] == "xor_bytes"]
    if len(raw_plan) + len(matrix_plan) != len(plan):
        raise ValueError("Unsupported fixture tensor encoding")

    for version in range(1, args.versions + 1):
        writer = PublicationWriter(args.output / codec / f"v{version}", stream_id=stream_id, base_version=version - 1, target_version=version, plan_digest=digest, publication_id=f"{stream_id}:{version}")
        started, changed = time.monotonic(), 0
        pending, batch, batch_bytes = [], [], 0
        try:
            for tensor in raw_plan:
                before, after = _alter_fixture_tensor(target, index, tensor, version=version, seed=args.seed, rate=rate)
                entry = writer.add_raw_tensor(tensor["name"], before, after, dtype=tensor["dtype"], shape=tensor["shape"], views=tensor["views"])
                changed += entry["changed_bytes"]
            for tensor in matrix_plan:
                before, after = _alter_fixture_tensor(target, index, tensor, version=version, seed=args.seed, rate=rate)
                # Pinned snapshots are bounded by a batching target, except that
                # one larger tensor remains whole. Only compact Snappy survives
                # each GPU batch; the canonical snapshots are then released.
                batch.append((torch.from_numpy(before.copy()).pin_memory(), torch.from_numpy(after).pin_memory(), "xor_bytes"))
                batch_bytes += after.nbytes
                if batch_bytes >= 512 * 1024**2:
                    pending.extend(encoder.encode_device(batch))
                    batch, batch_bytes = [], 0
            if batch:
                pending.extend(encoder.encode_device(batch))
            batch = []
            for tensor, (frames, payload, outer, count, _) in zip(matrix_plan, encoder.wrap_device(pending), strict=True):
                writer.add_gpu_outer_tensor(tensor["name"], frames, payload, outer, changed_bytes=count, dtype=tensor["dtype"], shape=tensor["shape"], views=tensor["views"])
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


async def _request(client, endpoint, payload=None, *, timeout=1200):
    async with httpx.AsyncClient(trust_env=False, timeout=timeout) as http:
        response = await (http.get(client.server_url + "/" + endpoint) if payload is None else http.post(client.server_url + "/" + endpoint, json=payload))
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
    # Keep fixture creation independent of serving client imports.
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
            code = "import json,sys; from sglang.srt.server_args import ServerArgs; from sglang.srt.entrypoints.http_server import launch_server; launch_server(ServerArgs(**json.load(open(sys.argv[1]))))"
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
                "feature_env": {key: os.environ.get(key) for key in ("WEIGHT_DELTA_CODEC", "WEIGHT_DELTA_TIMING")},
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
    codec = configured_codec()
    fixture_key = codec
    fixture = json.loads((args.fixture / "fixture.json").read_text()) if args.fixture else None
    if args.phase == "run":
        fixture_key = _validate_fixture_codec(fixture)
    model = Path(fixture["target_checkpoint"]) if args.phase == "oracle" else args.model
    async with _engines(args, model) as clients:
        if args.phase == "oracle":
            _save(args.output / "target-generation.json", await _generation(clients))
            return
        descriptions = await asyncio.gather(*[c.get_weights_delta_info(engine_id=f"engine-{i:05d}") for i, c in enumerate(clients)])
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
    parser.add_argument("phase", choices=("inventory", "fixture", "run", "oracle"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--versions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--ratio", type=float, default=0.002)
    parser.add_argument("--ports", type=int, nargs=1, default=(31000,))
    parser.add_argument("--startup-timeout", type=float, default=3600)
    args = parser.parse_args()
    args.model = args.model.resolve(strict=True)
    if args.phase == "fixture" and args.inventory is None:
        parser.error("fixture requires --inventory")
    if args.phase in {"run", "oracle"} and args.fixture is None:
        parser.error("run/oracle requires --fixture")
    if len(set(args.ports)) != 1 or any(not 0 < port < 65536 for port in args.ports):
        parser.error("--ports requires one valid TCP port")
    if args.versions < 1 or not 0 < args.ratio < 0.1:
        parser.error("versions must be positive and ratio in (0, 0.1)")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.phase == "fixture":
        _fixture(args)
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
