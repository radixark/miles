import asyncio
import json
import logging
import sys
from urllib.parse import quote

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.router.config import MilesRouterConfig
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)


def run_router(config: MilesRouterConfig):
    """
    Run the Miles router with the specified configuration.
    """
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("miles_router")
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-router")

    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(config, verbose=False)

    # Start the server
    uvicorn.run(miles_router.app, host=config.host, port=config.port, log_level="info")


class MilesRouter:
    def __init__(self, config: MilesRouterConfig, verbose=False):
        """Initialize the miles-router with SGLang router address"""
        self.config = config
        self.verbose = verbose

        self.app = FastAPI()
        self.app.router.on_startup.append(self._start_background_health_check)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> KV-cache capacity in tokens, read once from the worker when it registers
        self.worker_capacities: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=config.max_connections),
            timeout=httpx.Timeout(config.timeout),
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup all the HTTP routes except catch-all proxy"""
        # sglang-router api
        self.app.post("/workers")(self.create_worker)
        self.app.get("/workers")(self.get_workers)
        self.app.delete("/workers/{worker_id:path}")(self.delete_worker)
        # sglang-router <= 0.2.1 api
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/list_workers")(self.list_workers)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[miles-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            logger.debug(f"[miles-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.config.health_check_interval
        threshold = self.config.health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if url not in self.worker_request_counts:
                        continue
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[miles-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[miles-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        result = await self.do_proxy(request, path)
        return self.build_proxy_response(result)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Core proxy logic. Returns dict with request_body, response_body, status_code, headers."""
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        if body is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            content = await response.aread()
            return {
                "request_body": body,
                "response_body": content,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
        finally:
            self._finish_url(worker_url)

    def build_proxy_response(self, result: dict) -> Response:
        """Build HTTP response from proxy result."""
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "server", "date")
        }
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except Exception:
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        worker_url = await self._parse_worker_url(request)
        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        await self._register_worker(worker_url)
        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def remove_worker(self, request: Request):
        """Remove a worker from the router, using the same URL conventions as add_worker."""
        worker_url = await self._parse_worker_url(request)
        if worker_url is None:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        self._remove_worker_url(worker_url)
        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def create_worker(self, request: Request):
        """POST /workers with body {"url", "worker_type"}, as sgl-router takes it."""
        body = await request.body()
        payload = json.loads(body) if body else {}
        worker_url = payload.get("url")
        if not worker_url:
            return JSONResponse(status_code=400, content={"error": "url is required in the JSON body"})
        if payload.get("worker_type", "regular") != "regular":
            return JSONResponse(
                status_code=400, content={"error": "miles router serves only regular workers (no PD disaggregation)"}
            )

        await self._register_worker(worker_url)
        return {"status": "success", "worker_id": _worker_id(worker_url)}

    async def get_workers(self, request: Request):
        return {"workers": [{"id": _worker_id(url), "url": url} for url in self.worker_request_counts]}

    async def delete_worker(self, worker_id: str):
        # the path is percent-decoded before routing, so the id arrives as the url itself
        self._remove_worker_url(worker_id)
        return {"status": "success"}

    async def _register_worker(self, worker_url: str) -> None:
        if worker_url in self.worker_request_counts:
            return
        capacity = await self._fetch_worker_capacity(worker_url)
        self._add_worker_url(worker_url, capacity=capacity)

    async def _fetch_worker_capacity(self, worker_url: str) -> int | None:
        """KV-cache capacity of a worker in tokens, or None when the worker does not report it.

        SGLang reports `max_total_num_tokens` per data-parallel replica, so the worker's
        capacity is that times `dp_size`.
        """
        try:
            response = await self.client.get(f"{worker_url}/get_server_info", timeout=_CAPACITY_FETCH_TIMEOUT)
            response.raise_for_status()
            info = response.json()
        except Exception as e:
            logger.warning(f"[miles-router] Could not read capacity of {worker_url}: {e!r}")
            return None
        return _capacity_from_server_info(info)

    def _add_worker_url(self, worker_url: str, *, capacity: int | None = None) -> None:
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            self.dead_workers.discard(worker_url)
            if capacity is not None:
                self.worker_capacities[worker_url] = capacity
            logger.info(f"[miles-router] Worker {worker_url} KV capacity: {capacity} tokens")
            if self.verbose:
                print(f"[miles-router] Added new worker: {worker_url}")

    def _remove_worker_url(self, worker_url: str) -> None:
        self.worker_request_counts.pop(worker_url, None)
        self.worker_capacities.pop(worker_url, None)
        self.worker_failure_counts.pop(worker_url, None)
        self.dead_workers.discard(worker_url)
        logger.info(f"[miles-router] Removed worker: {worker_url}")

    async def _parse_worker_url(self, request: Request) -> str | None:
        if worker_url := request.query_params.get("url") or request.query_params.get("worker_url"):
            return worker_url

        body = await request.body()
        payload = json.loads(body) if body else {}
        return payload.get("url") or payload.get("worker_url")

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_request_counts.keys())}

    def _use_url(self):
        """Select the live worker with the least load (see `_select_worker`)."""
        live_workers = [w for w in self.worker_request_counts if w not in self.dead_workers]
        if not live_workers:
            raise RuntimeError("No healthy workers available in the pool")

        url = _select_worker(live_workers, self.worker_request_counts, self.worker_capacities)
        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url: str) -> None:
        """Mark the request to the given URL as finished"""
        if (count := self.worker_request_counts.get(url)) is None or count == 0:
            logger.info(f"[miles-router] Request to {url} finished after the worker was deregistered; ignoring")
            return
        self.worker_request_counts[url] = count - 1


_CAPACITY_FETCH_TIMEOUT = 30.0


def _capacity_from_server_info(info: dict) -> int | None:
    tokens_per_replica = info.get("max_total_num_tokens")
    if not tokens_per_replica:
        return None
    return int(tokens_per_replica) * max(int(info.get("dp_size") or 1), 1)


def _select_worker(workers: list[str], request_counts: dict[str, int], capacities: dict[str, int]) -> str:
    """Pick the worker with the fewest in-flight requests per token of KV capacity.

    Weighting by capacity keeps a mixed pool (e.g. a 2.7M-token B300 engine next to a 0.23M-token
    H200 engine) loaded in proportion to what each engine can hold, instead of giving every engine
    the same number of requests. `+ 1` counts the request being placed, so ties between idle
    workers go to the larger one. When any worker's capacity is unknown, all workers fall back to
    plain least-requests, so a partial capacity map never skews the pool.
    """
    if all(w in capacities for w in workers):
        return min(workers, key=lambda w: (request_counts[w] + 1) / capacities[w])
    return min(workers, key=request_counts.__getitem__)


def _worker_id(worker_url: str) -> str:
    # quoted so the id is one path segment of DELETE /workers/{id}
    return quote(worker_url, safe="")


if __name__ == "__main__":
    run_router(parse_config_argv(MilesRouterConfig, sys.argv[1:]))
