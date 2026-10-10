"""Manager-side `/samples` ingest benchmark: read one reply body and decode it, at a given concurrency.

Measures what `OpenAIEndpointTracer.collect_samples` costs the rollout manager per collected
sample, minus the session DELETE: the POST that reads the reply body, then
`decode_samples_and_merge_input_sample` in `asyncio.to_thread`. One asyncio loop in one process,
`--concurrency` requests in flight, `--bodies` replies in total. Two clients:

  buffer  `post_buffer_no_retry`, the production path (body read into one Content-Length buffer,
          tensor fields decoded as views of it)
  httpx   the earlier read: `httpx.AsyncClient.post(...).content` with production pool limits. Run
          it against a checkout from before `post_buffer_no_retry` (copy this file there) for the
          earlier read + decode; on this tree it decodes with the current codec

The server is a minimal asyncio responder in `--server-procs` processes on one SO_REUSEPORT port,
replying to every POST with one pre-encoded `encode_samples` payload and the session server's
reply headers (content-length, application/octet-stream), so its own cost stays off the
measurement. The default payload is 200 MB: one sample of 40k tokens with int32 R3 [T-1, 78, 16].

    python tests/manual/session/bench_samples_ingest.py --client buffer --concurrency 2048 --bodies 2048
    for c in httpx buffer; do for n in 1 128 2048; do
        python tests/manual/session/bench_samples_ingest.py --client $c --concurrency $n --bodies $((n > 64 ? n : 64))
    done; done

Prints one JSON line: bodies/s, GB/s, client CPU per body (user/sys), body wall p50/p99, peak
RSS, event-loop lag. A run aborts (exit 3) when node available memory drops below
`--mem-floor-gb`; in-flight bodies hold about one body size each with `buffer`, three with `httpx`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import resource
import signal
import socket
import tempfile
import threading
import time

# numpy, httpx, psutil and miles are imported in the functions that use them: each spawned server
# process re-imports this module and needs none of them.


def _serve(port: int, payload_path: str) -> None:
    """Spawn target: answer every POST on `port` with the payload, honoring keep-alive."""

    async def handle(loop, conn: socket.socket, head: bytes, body: memoryview) -> None:
        pending = b""
        try:
            while True:
                while b"\r\n\r\n" not in pending:
                    chunk = await loop.sock_recv(conn, 65536)
                    if not chunk:
                        return
                    pending += chunk
                request_head, _, rest = pending.partition(b"\r\n\r\n")
                lines = [line.lower() for line in request_head.split(b"\r\n")]
                length = next((int(line.split(b":")[1]) for line in lines if line.startswith(b"content-length")), 0)
                while len(rest) < length:
                    chunk = await loop.sock_recv(conn, 65536)
                    if not chunk:
                        return
                    rest += chunk
                pending = rest[length:]
                await loop.sock_sendall(conn, head)
                await loop.sock_sendall(conn, body)
                if b"connection: close" in lines:
                    return
        except OSError:
            return
        finally:
            conn.close()

    async def serve() -> None:
        with open(payload_path, "rb") as f:
            payload = f.read()
        head = f"HTTP/1.1 200 OK\r\ncontent-length: {len(payload)}\r\ncontent-type: application/octet-stream\r\n\r\n"
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(8192)
        listener.setblocking(False)
        loop = asyncio.get_running_loop()
        while True:
            conn, _ = await loop.sock_accept(listener)
            conn.setblocking(False)
            loop.create_task(handle(loop, conn, head.encode(), memoryview(payload)))

    asyncio.run(serve())


def _build_payload(tokens: int, layers: int, topk: int) -> bytes:
    import numpy as np

    from miles.rollout.session.samples.codec import encode_samples
    from miles.utils.types import Sample

    rng = np.random.default_rng(0)
    token_ids = rng.integers(0, 150_000, size=tokens).tolist()
    response_length = tokens // 2
    sample = Sample(
        tokens=token_ids,
        response="x" * response_length,
        response_length=response_length,
        loss_mask=[1] * response_length,
        rollout_log_probs=(-rng.random(response_length)).tolist(),
        rollout_routed_experts=rng.integers(1, 256, size=(tokens - 1, layers, topk), dtype=np.int32),
        status=Sample.Status.COMPLETED,
    )
    return encode_samples([sample], {"accumulated_token_ids": token_ids, "max_trim_tokens": 0})


def _start_server(payload: bytes, procs: int) -> tuple[list[multiprocessing.process.BaseProcess], int, str]:
    payload_file = tempfile.NamedTemporaryFile(prefix="miles-ingest-", suffix=".bin", delete=False)
    payload_file.write(payload)
    payload_file.close()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ctx = multiprocessing.get_context("spawn")
    servers = [ctx.Process(target=_serve, args=(port, payload_file.name), daemon=True) for _ in range(procs)]
    for server in servers:
        server.start()
    deadline = time.time() + 60
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return servers, port, payload_file.name
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)


async def _drive(args, url: str) -> tuple[list[float], list[float], dict[str, int]]:
    import httpx

    from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
    from miles.utils.types import Sample

    if args.client == "buffer":
        from miles.utils.http_utils import post_buffer_no_retry

    body = {"max_seq_len": None}
    walls: list[float] = []
    lags: list[float] = []
    errors: dict[str, int] = {}
    started = 0
    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency), timeout=httpx.Timeout(None))

    async def read() -> object:
        if args.client == "buffer":
            return await post_buffer_no_retry(url, body, timeout=args.timeout)
        response = await asyncio.wait_for(client.post(url, json=body), timeout=args.timeout)
        response.raise_for_status()
        return response.content

    async def worker() -> None:
        nonlocal started
        while started < args.bodies:
            started += 1
            t0 = time.perf_counter()
            try:
                reply = await asyncio.to_thread(decode_samples_and_merge_input_sample, await read(), Sample())
                del reply
                walls.append(time.perf_counter() - t0)
            except Exception as e:
                errors[type(e).__name__] = errors.get(type(e).__name__, 0) + 1

    async def lag_probe(stop: asyncio.Event) -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            await asyncio.sleep(0.05)
            lags.append(time.perf_counter() - t0 - 0.05)

    stop = asyncio.Event()
    probe = asyncio.create_task(lag_probe(stop))
    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    stop.set()
    await probe
    await client.aclose()
    return walls, lags, errors


def _watch_memory(floor_gb: float) -> None:
    import psutil

    def abort(signum, frame):
        raise SystemExit(3)

    signal.signal(signal.SIGUSR1, abort)

    def watch() -> None:
        while True:
            available = psutil.virtual_memory().available
            if available < floor_gb * 1e9:
                print(
                    json.dumps({"aborted": f"node available memory {available / 1e9:.0f} GB < {floor_gb} GB"}),
                    flush=True,
                )
                # Exit on the main thread so main's finally stops servers and removes their payload.
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            time.sleep(0.2)

    threading.Thread(target=watch, daemon=True).start()


def _pct(values: list[float], q: float) -> float | None:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Manager-side /samples ingest benchmark")
    parser.add_argument("--client", choices=["buffer", "httpx"], required=True)
    parser.add_argument("--concurrency", type=int, default=128, help="requests in flight")
    parser.add_argument("--bodies", type=int, default=128, help="replies to read in total")
    parser.add_argument("--tokens", type=int, default=40_000, help="sample length; R3 has tokens-1 rows")
    parser.add_argument("--layers", type=int, default=78, help="R3 layers")
    parser.add_argument("--topk", type=int, default=16, help="R3 experts per token and layer")
    parser.add_argument("--server-procs", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3600.0, help="per-request total timeout, seconds")
    parser.add_argument("--mem-floor-gb", type=float, default=64.0, help="abort below this node available memory")
    args = parser.parse_args()

    payload = _build_payload(args.tokens, args.layers, args.topk)
    servers, port, payload_path = _start_server(payload, args.server_procs)
    try:
        _watch_memory(args.mem_floor_gb)
        usage0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
        walls, lags, errors = asyncio.run(_drive(args, f"http://127.0.0.1:{port}/sessions/bench/samples"))
        wall, usage1 = time.perf_counter() - t0, resource.getrusage(resource.RUSAGE_SELF)
    finally:
        for server in servers:
            server.terminate()
        for server in servers:
            server.join()
        os.unlink(payload_path)
    done = max(len(walls), 1)
    print(
        json.dumps(
            {
                "client": args.client,
                "concurrency": args.concurrency,
                "body_mb": round(len(payload) / 1e6, 1),
                "completed": len(walls),
                "errors": errors,
                "wall_s": round(wall, 2),
                "bodies_per_s": round(len(walls) / wall, 2),
                "gb_per_s": round(len(walls) * len(payload) / wall / 1e9, 2),
                "cpu_user_ms_per_body": round(1e3 * (usage1.ru_utime - usage0.ru_utime) / done, 1),
                "cpu_sys_ms_per_body": round(1e3 * (usage1.ru_stime - usage0.ru_stime) / done, 1),
                "body_wall_s_p50": _pct(walls, 0.5),
                "body_wall_s_p99": _pct(walls, 0.99),
                "loop_lag_s_max": max(lags, default=None),
                "peak_rss_gb": round(usage1.ru_maxrss * 1024 / 1e9, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
