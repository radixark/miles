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
import json
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from tests.manual.gpu_delta.fixture_gpu_delta import _fixture
from tests.manual.gpu_delta.report_gpu_delta import write_report

from miles.backends.training_utils.weight_update.protocols.gpu_delta.session import (
    activate_publication,
    negotiate_cohort,
)
from miles.utils.gpu_delta.publication import CODECS, FRAME_BYTES, configured_codec

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
    write_report(args.output)
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
