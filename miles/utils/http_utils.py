import asyncio
import ipaddress
import json
import logging
import mmap
import multiprocessing
import os
import random
import socket
import subprocess
import time
import urllib.parse

import httpx
import numpy as np

from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)

MILES_HOST_IP_ENV = "MILES_HOST_IP"
MILES_PREFER_IPV6_ENV = "MILES_PREFER_IPV6"

_CONNECT_ATTEMPT_TIMEOUT_SECONDS = 1.0
_CONNECT_RETRY_INTERVAL_SECONDS = 0.5


def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def wait_for_server_ready(
    host: str,
    port: int,
    process: "multiprocessing.Process | subprocess.Popen | None" = None,
    timeout: float = 30,
) -> None:
    """Poll until a TCP port is accepting connections.

    Raises ``RuntimeError`` if the process dies or the timeout is exceeded.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and not _is_process_running(process):
            raise RuntimeError(f"Server process died before port {port} became ready")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Server at {host}:{port} not ready after {timeout}s")


async def wait_tcp_ready_async(host: str, port: int, *, timeout: float = 30) -> None:
    """Poll until a TCP port accepts connections."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host.strip("[]"), port), timeout=_CONNECT_ATTEMPT_TIMEOUT_SECONDS
            )
            writer.close()
            await writer.wait_closed()
            return
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(_CONNECT_RETRY_INTERVAL_SECONDS)
    raise RuntimeError(f"Server at {host}:{port} not ready after {timeout}s")


def _is_process_running(process: "multiprocessing.Process | subprocess.Popen") -> bool:
    if isinstance(process, subprocess.Popen):
        return process.poll() is None
    return process.is_alive()


def resolve_ip(host: str) -> str:
    bare = host.strip("[]")
    try:
        ipaddress.ip_address(bare)
    except ValueError:
        pass
    else:
        return wrap_ipv6(bare)

    # "Prefer" is indeed strict to match old semantics
    prefer_ipv6 = os.getenv(MILES_PREFER_IPV6_ENV, "0").lower() in ("1", "true", "yes", "on")
    family = socket.AF_INET6 if prefer_ipv6 else socket.AF_INET
    # getaddrinfo allows specifying the family (AF_INET or AF_INET6)
    # Result format: [(family, type, proto, canonname, sockaddr), ...]
    infos = socket.getaddrinfo(bare, None, family=family, type=socket.SOCK_STREAM)
    for info in infos:
        ip = info[4][0]  # The first element of sockaddr is the IP
        # Must filter out loopback addresses to avoid "127.0.0.1" issues
        if not ipaddress.ip_address(ip).is_loopback:
            return wrap_ipv6(ip)

    raise RuntimeError(f"{host!r} resolves to no non-loopback {family.name} address, so nothing can bind on it")


def wrap_ipv6(host):
    """Wrap IPv6 address in [] if needed."""
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        return host


def router_worker_base_urls(urls: list[str]) -> list[str]:
    """Strip the `@<rank>` suffix dp-aware routing adds; ranks of one engine dedupe to one address."""
    bases = []
    for url in urls:
        base, sep, rank = url.rpartition("@")
        if sep and rank.isdigit():
            url = base
        if url not in bases:
            bases.append(url)
    return bases


def run_router(args):
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("router")
    try:
        from sglang_router.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception as e:
        logger.info(e)
        return 1


def terminate_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    """Terminate a process gracefully, with forced kill as fallback.

    Args:
        process: The process to terminate
        timeout: Seconds to wait for graceful termination before forcing kill
    """
    if not process.is_alive():
        return

    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()


class GeneralHttpClientProvider:
    _CONNECT_TIMEOUT = 10.0
    _WRITE_TIMEOUT = 60.0
    _POOL_TIMEOUT = 60.0
    _TIMEOUT = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=None, write=_WRITE_TIMEOUT, pool=_POOL_TIMEOUT)
    _LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=None)

    # TODO: entries are never evicted and the clients are never aclose()d, so a caller that keeps
    # creating event loops (repeated asyncio.run) leaks one client and its keep-alive sockets per
    # loop. Today's call sites use a bounded number of loops; add eviction before that stops holding.
    _clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}

    @classmethod
    def client(cls) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        client = cls._clients.get(loop)
        if client is None:
            client = httpx.AsyncClient(timeout=cls._TIMEOUT, limits=cls._LIMITS)
            cls._clients[loop] = client
        return client


# TODO: the client below is not general — it carries a rollout-specific connection limit and an
# optional ray-distributed POST path. Rename it (or fold it into GeneralHttpClientProvider with the
# limit as an argument) once the rollout request path is reworked.
_http_client: httpx.AsyncClient | None = None
_client_concurrency: int = 0

# Optional Ray-based distributed POST dispatch
_distributed_post_enabled: bool = False
_post_actors: list[object] = []
_post_actor_idx: int = 0


def _next_actor():
    global _post_actor_idx
    if not _post_actors:
        return None
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    return actor


async def _post(client, url, payload, max_retries=60, action="post", headers=None):
    retry_count = 0
    while retry_count < max_retries:
        try:
            if action in ("delete", "get"):
                assert not payload
                response = await getattr(client, action)(url, headers=headers)
            else:
                response = await getattr(client, action)(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            try:
                output = response.json()
            except json.JSONDecodeError:
                output = response.text
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output


async def wait_http_ok(url: str, *, json_payload=None, timeout: float = 180.0, request_timeout: float = 60.0) -> None:
    """Poll ``url`` until it answers HTTP 200 (POST when ``json_payload`` is given,
    else GET); raise ``TimeoutError`` past the deadline."""
    deadline = time.time() + timeout
    last_error = "no attempt made"
    async with httpx.AsyncClient() as client:
        while True:
            try:
                if json_payload is not None:
                    response = await client.post(url, json=json_payload, timeout=request_timeout)
                else:
                    response = await client.get(url, timeout=request_timeout)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as e:
                last_error = repr(e)
            if time.time() > deadline:
                raise TimeoutError(f"{url} not ready after {timeout}s: {last_error}")
            await asyncio.sleep(5)


async def post_buffer_no_retry(url: str, payload: dict, *, timeout: float) -> np.ndarray:
    """Perform one JSON POST and return the whole reply body as a fresh writable uint8 array.

    The body is read with ``sock_recv_into`` straight into one buffer sized by the reply's
    Content-Length. A bulk reply (a ``/samples`` body is mostly R3 and can be hundreds of MB)
    read through httpx is copied several times and parsed 64 KiB at a time on the event loop.
    Plain ``http://`` only, one connection per call, no retry, total ``timeout``. Transport
    failures raise ``httpx.TransportError`` subclasses and a non-2xx reply raises
    ``RuntimeError`` carrying its body, as the httpx client did.
    """
    return await asyncio.wait_for(_post_buffer(url, payload), timeout=timeout)


_MAX_REPLY_HEAD_BYTES = 64 * 1024


async def _post_buffer(url: str, payload: dict) -> np.ndarray:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http":
        raise ValueError(f"post_buffer_no_retry supports http:// only, got {url}")
    body = json.dumps(payload).encode()
    request = (
        f"POST {parts.path or '/'}{'?' + parts.query if parts.query else ''} HTTP/1.1\r\n"
        f"Host: {parts.netloc}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + body
    loop = asyncio.get_running_loop()
    sock = await _connect_socket(loop, parts.hostname, parts.port or 80)
    try:
        status, length, received = await _send_and_read_reply_head(loop, sock, request)
        # A private anonymous mapping, not np.empty: numpy madvises large allocations to huge pages,
        # and on a fragmented node each such allocation stalls in direct compaction (2048 bodies of
        # 200 MB in flight: 8.4 -> 2.3 bodies/s). mmap rejects length 0.
        mapping = mmap.mmap(-1, max(length, 1), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        reply = np.frombuffer(mapping, dtype=np.uint8)[:length]
        await _recv_exactly_into(loop, sock, reply, received)
    finally:
        sock.close()
    if not (200 <= status < 300):
        raise RuntimeError(f"POST {url} failed with {status}: {reply.tobytes().decode(errors='replace')}")
    return reply


async def _connect_socket(loop: asyncio.AbstractEventLoop, host: str, port: int) -> socket.socket:
    try:
        family, sock_type, proto, _, addr = (await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM))[0]
        sock = socket.socket(family, sock_type, proto)
        sock.setblocking(False)
        try:
            await loop.sock_connect(sock, addr)
        except BaseException:
            sock.close()
            raise
    except OSError as e:
        raise httpx.ConnectError(f"connect to {host}:{port} failed: {e!r}") from e
    return sock


async def _send_and_read_reply_head(
    loop: asyncio.AbstractEventLoop, sock: socket.socket, request: bytes
) -> tuple[int, int, bytes]:
    """Send the request; return (status, Content-Length, body bytes already received)."""
    try:
        await loop.sock_sendall(sock, request)
    except OSError as e:
        raise httpx.WriteError(repr(e)) from e
    head = b""
    try:
        while b"\r\n\r\n" not in head:
            if len(head) > _MAX_REPLY_HEAD_BYTES:
                raise httpx.RemoteProtocolError(f"reply head exceeds {_MAX_REPLY_HEAD_BYTES} bytes")
            chunk = await loop.sock_recv(sock, 65536)
            if not chunk:
                raise httpx.RemoteProtocolError("connection closed before the reply head")
            head += chunk
    except OSError as e:
        raise httpx.ReadError(repr(e)) from e
    head, _, received = head.partition(b"\r\n\r\n")
    status_line, *header_lines = head.split(b"\r\n")
    headers = {
        name.strip().lower(): value.strip() for name, _, value in (line.partition(b":") for line in header_lines)
    }
    try:
        status = int(status_line.split()[1])
        length = int(headers[b"content-length"])
    except (IndexError, KeyError, ValueError) as e:
        raise httpx.RemoteProtocolError(f"reply head without a status or Content-Length: {head[:200]!r}") from e
    if len(received) > length:
        raise httpx.RemoteProtocolError(f"reply carries {len(received)} bytes past a Content-Length of {length}")
    return status, length, received


async def _recv_exactly_into(
    loop: asyncio.AbstractEventLoop, sock: socket.socket, buffer: np.ndarray, received: bytes
) -> None:
    view = memoryview(buffer)
    view[: len(received)] = received
    filled = len(received)
    try:
        while filled < len(buffer):
            n = await loop.sock_recv_into(sock, view[filled:])
            if n == 0:
                raise httpx.ReadError(f"connection closed after {filled} of {len(buffer)} body bytes")
            filled += n
    except OSError as e:
        raise httpx.ReadError(repr(e)) from e


def init_http_client(args):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency, _distributed_post_enabled
    rollout_num_gpus = args.rollout_num_gpus or 0
    if rollout_num_gpus == 0 and not args.eval_uses_snapshots:
        return

    _client_concurrency = args.sglang_server_concurrency * rollout_num_gpus // args.rollout_num_gpus_per_engine
    if args.eval_num_gpus > 0:
        _client_concurrency += args.sglang_server_concurrency * args.eval_num_gpus // args.eval_num_gpus_per_engine
    _client_concurrency = max(_client_concurrency, args.sglang_server_concurrency)
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_client_concurrency),
            timeout=httpx.Timeout(None),
        )

    # Optionally initialize distributed POST via Ray without changing interfaces
    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _init_ray_distributed_post(args):
    """Initialize one or more Ray async actors per node for HTTP POST.

    Uses NodeAffinitySchedulingStrategy to place actors on distinct nodes.
    Controlled by MILES_HTTP_POST_ACTORS_PER_NODE.
    """
    global _post_actors
    if _post_actors:
        return  # Already initialized

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Discover alive nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

    # Define the async actor
    @ray.remote
    class _HttpPosterActor:
        def __init__(self, *, concurrency: int):
            # Lazy creation to this actor's event loop
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max(1, concurrency)),
                timeout=httpx.Timeout(None),
            )

        async def do_post(self, url, payload, max_retries=60, action="post", headers=None):
            return await _post(self._client, url, payload, max_retries, action=action, headers=headers)

    # Create actors per node
    created = []
    # Distribute client concurrency across actors (at least 1 per actor)
    per_actor_conc = (_client_concurrency + len(nodes)) // len(nodes)

    for node in nodes:
        node_id = node["NodeID"]
        scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        for _ in range(args.num_gpus_per_node):
            actor = _HttpPosterActor.options(
                name=None,
                lifetime="detached",
                scheduling_strategy=scheduling,
                max_concurrency=per_actor_conc,
                # Use tiny CPU to schedule
                num_cpus=0.001,
            ).remote(concurrency=per_actor_conc)
            created.append(actor)

    _post_actors = created


def _rollout_client() -> httpx.AsyncClient:
    # a process where init_http_client never ran, such as a subproc agent call, gets a default client
    return _http_client if _http_client is not None else GeneralHttpClientProvider.client()


# TODO may generalize the name since it now contains http DELETE/GET etc (with retries and remote-execution)
async def post(url, payload, max_retries=60, action="post", headers=None):
    # If distributed mode is enabled and actors exist, dispatch via Ray.
    if _distributed_post_enabled and _post_actors:
        try:
            actor = _next_actor()
            if actor is not None:
                return await actor.do_post.remote(url, payload, max_retries, action=action, headers=headers)
        except Exception as e:
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")
            # fall through to local

    return await _post(_rollout_client(), url, payload, max_retries, action=action, headers=headers)


# TODO unify w/ `post` to add retries and remote-execution
async def get(url):
    response = await _rollout_client().get(url)
    response.raise_for_status()
    output = response.json()
    return output
