import asyncio
import ipaddress
import json
import logging
import multiprocessing
import os
import random
import socket
import time

import aiohttp

logger = logging.getLogger(__name__)

MILES_HOST_IP_ENV = "MILES_HOST_IP"


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
    process: "multiprocessing.Process | None" = None,
    timeout: float = 30,
) -> None:
    """Poll until a TCP port is accepting connections.

    Raises ``RuntimeError`` if the process dies or the timeout is exceeded.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and not process.is_alive():
            raise RuntimeError(f"Server process died before port {port} became ready")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Server at {host}:{port} not ready after {timeout}s")


def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(MILES_HOST_IP_ENV, None):
        return hostname, env_overwrite_local_ip

    def _is_loopback(ip):
        return ip.startswith("127.") or ip == "::1"

    def _resolve_ip(family, test_target_ip):
        """
        Attempt to get the local LAN IP for the specific family (IPv4/IPv6).
        Strategy: UDP Probe (Preferred) -> Hostname Resolution (Fallback) -> None
        """

        # Strategy 1: UDP Connect Probe (Most accurate, relies on routing table)
        # Useful when the machine has a default gateway or internet access.
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                # The IP doesn't need to be reachable, but the routing table must exist.
                s.connect((test_target_ip, 80))
                ip = s.getsockname()[0]
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass  # Route unreachable or network error, move to next strategy.

        # Strategy 2: Hostname Resolution (Fallback for offline clusters)
        # Useful for offline environments where UDP connect fails but /etc/hosts is configured.
        try:
            # getaddrinfo allows specifying the family (AF_INET or AF_INET6)
            # Result format: [(family, type, proto, canonname, sockaddr), ...]
            infos = socket.getaddrinfo(hostname, None, family=family, type=socket.SOCK_STREAM)

            for info in infos:
                ip = info[4][0]  # The first element of sockaddr is the IP
                # Must filter out loopback addresses to avoid "127.0.0.1" issues
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass

        return None

    prefer_ipv6 = os.getenv("MILES_PREFER_IPV6", "0").lower() in ("1", "true", "yes", "on")
    local_ip = None
    final_fallback = "127.0.0.1"

    if prefer_ipv6:
        # [Strict Mode] IPv6 Only
        # 1. Try UDP V6 Probe
        # 2. Try Hostname Resolution (V6)
        # If failed, fallback to V6 loopback. Never mix with V4.
        local_ip = _resolve_ip(socket.AF_INET6, "2001:4860:4860::8888")
        final_fallback = "::1"
    else:
        # [Strict Mode] IPv4 Only (Default)
        # 1. Try UDP V4 Probe
        # 2. Try Hostname Resolution (V4)
        # If failed, fallback to V4 loopback. Never mix with V6.
        local_ip = _resolve_ip(socket.AF_INET, "8.8.8.8")
        final_fallback = "127.0.0.1"

    return hostname, local_ip or final_fallback


def _wrap_ipv6(host):
    """Wrap IPv6 address in [] if needed."""
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        return host


def run_router(args):
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


_http_session: aiohttp.ClientSession | None = None
_client_concurrency: int = 0

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


async def _request(session, url, payload, max_retries=60, action="post"):
    retry_count = 0
    while retry_count < max_retries:
        try:
            kwargs = {} if action in ("delete", "get") else {"json": payload or {}}
            async with session.request(action, url, **kwargs) as response:
                response.raise_for_status()
                try:
                    output = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    output = await response.text()
        except Exception as e:
            retry_count += 1
            response_text = e.message if isinstance(e, aiohttp.ClientResponseError) else None
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


def init_http_client(args):
    """Initialize aiohttp session and optionally enable distributed POST via Ray."""
    global _http_session, _client_concurrency, _distributed_post_enabled
    if not args.rollout_num_gpus:
        return

    _client_concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    if _http_session is None:
        _http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=_client_concurrency, limit_per_host=_client_concurrency),
            timeout=aiohttp.ClientTimeout(total=None),
        )

    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _init_ray_distributed_post(args):
    """Initialize Ray async actors on the current node for HTTP POST.

    Pins all actors to the current node to avoid cross-node network hops.
    Each actor gets its own aiohttp session with a right-sized connection pool.
    """
    global _post_actors
    if _post_actors:
        return

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    current_node_id = ray.get_runtime_context().get_node_id()
    num_actors = args.num_gpus_per_node
    per_actor_conc = max(1, _client_concurrency // num_actors)

    @ray.remote
    class _HttpPosterActor:
        def __init__(self, concurrency: int):
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency),
                timeout=aiohttp.ClientTimeout(total=None),
            )

        async def do_post(self, url, payload, max_retries=60, action="post"):
            return await _request(self._session, url, payload, max_retries, action=action)

    scheduling = NodeAffinitySchedulingStrategy(node_id=current_node_id, soft=False)
    created = []
    for _ in range(num_actors):
        actor = _HttpPosterActor.options(
            name=None,
            lifetime="detached",
            scheduling_strategy=scheduling,
            max_concurrency=per_actor_conc,
            num_cpus=0.001,
        ).remote(per_actor_conc)
        created.append(actor)

    _post_actors = created


async def post(url, payload, max_retries=60, action="post"):
    if _distributed_post_enabled and _post_actors:
        try:
            actor = _next_actor()
            if actor is not None:
                return await actor.do_post.remote(url, payload, max_retries, action=action)
        except Exception as e:
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")

    return await _request(_http_session, url, payload, max_retries, action=action)


async def get(url):
    async with _http_session.get(url) as response:
        response.raise_for_status()
        return await response.json()
