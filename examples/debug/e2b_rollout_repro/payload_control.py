"""Controlled session payload replay; no optimizer or model weight changes."""

import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import psutil
import uvicorn
import uvloop
from e2b import AsyncSandbox
from fastapi import FastAPI, Request, Response
from tap import Tap

import miles.rollout.session.core as session_core
from miles.rollout.session.config import compute_session_server_config
from miles.rollout.session.server import SessionServer
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.utils.types import Sample


class Args(Tap):
    root: str
    parent: str
    mode: str = "driver"
    arm: str = "full"
    worker: int = 0
    workers: int = 8
    concurrency: int = 128
    repetitions: int = 2
    port: int = 33400
    engine: str = ""
    sandbox_count: int = 128


def without_candidates(response: dict) -> dict:
    result = copy.deepcopy(response)
    for choice in result["choices"]:
        choice.get("meta_info", {}).pop("output_top_logprobs", None)
        for token in (choice.get("logprobs") or {}).get("content", []):
            token.pop("top_logprobs", None)
    return result


async def heartbeat(stats: dict) -> None:
    proc = psutil.Process()
    while True:
        before = time.monotonic()
        await asyncio.sleep(0.1)
        stats["lag_s"].append(max(0, time.monotonic() - before - 0.1))
        stats["rss_max"] = max(stats["rss_max"], proc.memory_info().rss)
        stats["cpu_s"] = sum(proc.cpu_times()[:2])


def serve(args: Args) -> None:
    root = Path(args.root)
    fixture = json.loads((root / "fixture.json").read_text())
    if args.mode == "backend":
        app = FastAPI()
        full = json.dumps(fixture["response"]).encode()
        plain = json.dumps(without_candidates(fixture["response"])).encode()

        @app.post("/v1/chat/completions")
        async def completion(request: Request) -> Response:
            body = await request.json()
            assert body["input_ids"] == fixture["response"]["choices"][0]["prompt_token_ids"]
            return Response(full if body.get("top_logprobs") else plain, media_type="application/json")

        port = args.port - 1
    else:
        saved = json.loads((Path(args.parent) / "preflight-result.json").read_text())["args"]
        saved.update(loss_type="policy_loss" if args.arm == "plain" else "score_centering",
                     save_debug_trajectory_data=None)
        config = compute_session_server_config(
            SimpleNamespace(**saved), host="127.0.0.1", port=args.port + args.worker,
            instance_id=f"control-{args.arm}-{args.worker}", backend_url=f"http://127.0.0.1:{args.port - 1}",
        )
        if args.arm == "strip":
            original = session_core._strip_replay_payloads

            def strip_training_metadata(response: dict) -> dict:
                result = original(response)
                choices = []
                for choice in result["choices"]:
                    meta = {key: value for key, value in choice.get("meta_info", {}).items()
                            if key != "output_top_logprobs"}
                    choices.append({**choice, "meta_info": meta, "logprobs": None})
                return {**result, "choices": choices}

            session_core._strip_replay_payloads = strip_training_metadata
        app = SessionServer(config).app
        port = args.port + args.worker
    stats = {"lag_s": [], "rss_max": 0, "cpu_s": 0}

    @app.on_event("startup")
    async def start_monitor() -> None:
        app.state.monitor = asyncio.create_task(heartbeat(stats))

    @app.get("/control_metrics")
    async def metrics() -> dict:
        return stats

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def start_child(args: Args, mode: str, arm: str, worker: int, label: str) -> subprocess.Popen:
    command = [sys.executable, __file__, "--root", args.root, "--parent", args.parent,
               "--mode", mode, "--arm", arm, "--worker", str(worker), "--port", str(args.port)]
    with (Path(args.root) / f"{label}-{worker}.log").open("w") as stream:
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)


async def ready(client: httpx.AsyncClient, ports: list[int], processes: list[subprocess.Popen]) -> None:
    deadline = time.monotonic() + 240
    for port in ports:
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("Test child exited; inspect its log")
            try:
                response = await client.get(f"http://127.0.0.1:{port}/control_metrics", timeout=2)
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Test servers did not become ready")
                await asyncio.sleep(1)


def stop_children(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def prepare_fixture(args: Args, client: httpx.AsyncClient) -> None:
    root = Path(args.root)
    if (root / "fixture.json").exists():
        return
    parent = Path(args.parent)
    saved = json.loads((parent / "preflight-result.json").read_text())["args"]
    task = json.loads((parent / "dataset/train.jsonl").read_text().splitlines()[0])
    request = {"model": os.environ["AGENT_MODEL_NAME"].removeprefix("openai/"),
               "messages": task["prompt"], "max_tokens": 16384, "temperature": 1.0,
               "top_p": 1.0, "top_k": -1, "seed": 20260925, "logprobs": True,
               "top_logprobs": 128, "return_meta_info": True, "return_token_ids": True,
               "stream": False, "chat_template_kwargs": saved["apply_chat_template_kwargs"]}
    response = await client.post(args.engine + "/v1/chat/completions", json=request)
    response.raise_for_status()
    fixture = {"request": request, "response": response.json(), "task": task["label"]}
    (root / "fixture.json").write_text(json.dumps(fixture))
    print("FIXTURE", len(response.content), fixture["response"]["usage"], flush=True)


async def create_sandboxes(args: Args) -> list:
    templates = json.loads((Path(args.parent) / "template-manifest-resolved.json").read_text())["results"]
    template = next(row for row in templates if row["task"] == "fix-git")
    semaphore = asyncio.Semaphore(8)
    sandboxes = []

    async def create_one(index: int) -> None:
        async with semaphore:
            sandbox = await AsyncSandbox.create(
                template=template["template_id"], timeout=7200,
                metadata={"purpose": "session-payload-control", "run": Path(args.root).name, "slot": str(index)},
                api_key=Path(os.environ["E2B_API_KEY_FILE"]).read_text().strip(),
                api_url=os.environ["E2B_API_URL"],
            )
            sandboxes.append(sandbox)
            with (Path(args.root) / "owned-sandboxes.jsonl").open("a") as stream:
                stream.write(json.dumps({"id": sandbox.sandbox_id, "slot": index}) + "\n")

    try:
        results = await asyncio.gather(*(create_one(index) for index in range(args.sandbox_count)), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError(f"Sandbox creation failed: {failures[0]}")
    except BaseException:
        await asyncio.gather(*(sandbox.kill() for sandbox in sandboxes), return_exceptions=True)
        raise
    return sandboxes


async def one_trial(client: httpx.AsyncClient, args: Args, index: int, fixture: dict, sandboxes: list, arm: str) -> dict:
    base = f"http://127.0.0.1:{args.port + index % args.workers}"
    created = await client.post(base + "/sessions")
    created.raise_for_status()
    session = base + "/sessions/" + created.json()["session_id"]
    request = {key: value for key, value in fixture["request"].items()
               if key not in ("top_logprobs", "logprobs", "return_meta_info", "return_token_ids")}
    start = time.monotonic()
    row = {"index": index}
    try:
        response = await client.post(session + "/v1/chat/completions", json=request)
        response.raise_for_status()
        parsed = response.json()
        assert parsed["choices"][0]["message"] == fixture["response"]["choices"][0]["message"]
        row.update(chat_s=time.monotonic() - start, reply_bytes=len(response.content))
        del parsed, response
        if sandboxes:
            before = time.monotonic()
            try:
                result = await sandboxes[index % len(sandboxes)].commands.run("printf payload-control-ok", timeout=60)
                assert result.exit_code == 0 and result.stdout == "payload-control-ok"
                row["sandbox_ok"] = True
            except Exception as exc:
                row.update(sandbox_ok=False, sandbox_error=f"{type(exc).__name__}: {exc}")
            row["sandbox_s"] = time.monotonic() - before
        before = time.monotonic()
        response = await client.post(session + "/samples", json={"max_seq_len": 65536})
        response.raise_for_status()
        row.update(samples_s=time.monotonic() - before, sample_bytes=len(response.content))
        decoded = decode_samples_and_merge_input_sample(response.content, Sample(), fields=COMPUTED_FIELDS_V2)
        assert len(decoded.samples) == 1, decoded.empty_reason
        sample = decoded.samples[0]
        row["training_hash"] = hashlib.sha256(json.dumps([sample.tokens, sample.loss_mask, sample.rollout_log_probs]).encode()).hexdigest()
        candidates = sample.rollout_topk_log_probs
        assert (candidates is None) == (arm == "plain")
        if candidates is not None:
            row["candidate_hash"] = hashlib.sha256(candidates.tobytes() + sample.rollout_topk_token_ids.tobytes()).hexdigest()
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        response = await client.delete(session)
        row["delete_status"] = response.status_code
    return row


async def run_arm(client: httpx.AsyncClient, args: Args, arm: str, block: int, fixture: dict, sandboxes: list) -> dict:
    processes = [start_child(args, "server", arm, worker, f"{block}-{arm}") for worker in range(args.workers)]
    try:
        await ready(client, list(range(args.port, args.port + args.workers)), processes)
        before = [psutil.Process(process.pid).cpu_times() for process in processes]
        driver_stats = {"lag_s": [], "rss_max": 0, "cpu_s": 0}
        monitor = asyncio.create_task(heartbeat(driver_stats))
        start = time.monotonic()
        rows = await asyncio.gather(*(one_trial(client, args, index, fixture, sandboxes, arm) for index in range(args.concurrency)))
        duration = time.monotonic() - start
        await asyncio.sleep(0.15)
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        metrics = [(await client.get(f"http://127.0.0.1:{args.port + i}/control_metrics")).json()
                   for i in range(args.workers)]
        cpu = sum(sum(psutil.Process(p.pid).cpu_times()[:2]) - sum(b[:2]) for p, b in zip(processes, before))
        result = {"arm": arm, "block": block, "duration_s": duration, "server_cpu_s": cpu,
                  "metrics": metrics, "driver_metrics": driver_stats, "rows": rows}
        (Path(args.root) / f"result-{block}-{arm}.json").write_text(json.dumps(result))
        print("ARM_DONE", arm, block, duration, "errors", sum("error" in row for row in rows), flush=True)
        return result
    finally:
        stop_children(processes)


async def main_async(args: Args) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    sandboxes = []
    backend = None
    async with httpx.AsyncClient(timeout=10800, limits=httpx.Limits(max_connections=512), trust_env=False) as client:
        await prepare_fixture(args, client)
        fixture = json.loads((root / "fixture.json").read_text())
        (root / "fixture.sha256").write_text(hashlib.sha256((root / "fixture.json").read_bytes()).hexdigest())
        try:
            sandboxes = await create_sandboxes(args)
            backend = start_child(args, "backend", "full", 0, "backend")
            await ready(client, [args.port - 1], [backend])
            order = ["plain", "full", "strip", "strip", "full", "plain"][:args.repetitions * 3]
            results = [await run_arm(client, args, arm, i, fixture, sandboxes) for i, arm in enumerate(order)]
            (root / "results.json").write_text(json.dumps(results))
        finally:
            cleanup = await asyncio.gather(*(sandbox.kill() for sandbox in sandboxes), return_exceptions=True)
            (root / "cleanup.json").write_text(json.dumps([str(value) for value in cleanup]))
            if backend is not None:
                stop_children([backend])


def main() -> None:
    args = Args().parse_args()
    if args.mode == "driver":
        uvloop.run(main_async(args))
    else:
        serve(args)


if __name__ == "__main__":
    main()
