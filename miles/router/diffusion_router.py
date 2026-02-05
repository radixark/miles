import argparse
import asyncio
import json
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

logger = logging.getLogger(__name__)


def run_diffusion_router(args):
    """Run the diffusion router with the specified configuration."""
    router = DiffusionRouter(args)
    uvicorn.run(router.app, host=args.host, port=args.port, log_level="info")


class DiffusionRouter:
    def __init__(self, args, verbose=False):
        """Initialize the diffusion router for load-balancing across sglang-diffusion workers."""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()
        self.app.add_event_handler("startup", self._start_background_health_check)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()

        max_connections = getattr(args, "max_connections", 100)
        timeout = getattr(args, "timeout", None)

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup all the HTTP routes."""
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.get("/health")(self.health)
        self.app.get("/health_workers")(self.health_workers)
        self.app.post("/generate")(self.generate)
        self.app.post("/generate_video")(self.generate_video)
        self.app.post("/update_weights_from_disk")(self.update_weights_from_disk)
        # Catch-all route for proxying — must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    # ── Health checks ────────────────────────────────────────────────

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[diffusion-router] Worker {url} unhealthy (status {response.status_code})")
        except Exception as e:
            logger.debug(f"[diffusion-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and quarantine failing workers."""
        interval = getattr(self.args, "health_check_interval", 10)
        threshold = getattr(self.args, "health_check_failure_threshold", 3)

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures
                        if failures >= threshold:
                            logger.warning(
                                f"[diffusion-router] Worker {url} failed {threshold} consecutive checks. Marking DEAD."
                            )
                            self.dead_workers.add(url)
                            # Dead workers are permanently excluded. Reconnecting them
                            # would risk serving stale weights after training has moved on.
                    else:
                        self.worker_failure_counts[url] = 0

                healthy = len(self.worker_request_counts) - len(self.dead_workers)
                logger.debug(f"[diffusion-router] Health check complete. {healthy} workers healthy.")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[diffusion-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    # ── Load balancing ───────────────────────────────────────────────

    def _use_url(self):
        """Select worker URL with minimal active requests."""
        if not self.worker_request_counts:
            raise RuntimeError("No workers registered in the pool")
        if not self.dead_workers:
            url = min(self.worker_request_counts, key=self.worker_request_counts.get)
        else:
            valid_workers = (w for w in self.worker_request_counts if w not in self.dead_workers)
            try:
                url = min(valid_workers, key=self.worker_request_counts.get)
            except ValueError:
                raise RuntimeError("No healthy workers available in the pool") from None

        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url):
        """Mark the request to the given URL as finished."""
        assert url in self.worker_request_counts, f"URL {url} not recognized"
        self.worker_request_counts[url] -= 1
        assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"

    # ── Proxy helpers ────────────────────────────────────────────────

    async def _forward_to_worker(self, request: Request, path: str) -> Response:
        """Forward a request to the least-loaded worker and return the response."""
        try:
            worker_url = self._use_url()
        except RuntimeError as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})

        # TODO: Support streaming responses; current implementation buffers full response.
        query = request.url.query
        url = f"{worker_url}/{path}" if not query else f"{worker_url}/{path}?{query}"
        body = await request.body()
        headers = dict(request.headers)

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            content = await response.aread()
        finally:
            self._finish_url(worker_url)

        resp_headers = self._sanitize_response_headers(response.headers)
        content_type = resp_headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=response.status_code, headers=resp_headers)
        except Exception:
            return Response(
                content=content, status_code=response.status_code, headers=resp_headers, media_type=content_type
            )

    async def _broadcast_to_workers(self, path: str, body: bytes, headers: dict) -> list[dict]:
        """Send a request to ALL healthy workers and collect results."""
        urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
        if not urls:
            return []

        async def _send(worker_url):
            try:
                response = await self.client.post(f"{worker_url}/{path}", content=body, headers=headers)
                content = await response.aread()
                return {"worker_url": worker_url, "status_code": response.status_code, "body": json.loads(content)}
            except Exception as e:
                return {"worker_url": worker_url, "status_code": 502, "body": {"error": str(e)}}

        return await asyncio.gather(*(_send(u) for u in urls))

    @staticmethod
    def _sanitize_response_headers(headers) -> dict:
        """Remove hop-by-hop and encoding headers that no longer match buffered content."""
        hop_by_hop = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
                      "transfer-encoding", "upgrade"}
        dropped = {"content-length", "content-encoding"}
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop | dropped}

    # ── Route handlers ───────────────────────────────────────────────

    async def generate(self, request: Request):
        """Route image generation to the least-loaded worker via /v1/images/generations."""
        return await self._forward_to_worker(request, "v1/images/generations")

    async def generate_video(self, request: Request):
        """Route video generation to the least-loaded worker via /v1/videos/generations."""
        return await self._forward_to_worker(request, "v1/videos/generations")

    async def health(self, request: Request):
        """Aggregated health status: healthy if at least one worker is alive."""
        total = len(self.worker_request_counts)
        dead = len(self.dead_workers)
        healthy = total - dead
        status = "healthy" if healthy > 0 else "unhealthy"
        code = 200 if healthy > 0 else 503
        return JSONResponse(
            status_code=code,
            content={"status": status, "healthy_workers": healthy, "total_workers": total},
        )

    async def health_workers(self, request: Request):
        """Per-worker health and load information."""
        workers = []
        for url, count in self.worker_request_counts.items():
            workers.append({
                "url": url,
                "active_requests": count,
                "is_dead": url in self.dead_workers,
                "consecutive_failures": self.worker_failure_counts.get(url, 0),
            })
        return JSONResponse(content={"workers": workers})

    async def update_weights_from_disk(self, request: Request):
        """Broadcast weight reload to all healthy workers."""
        body = await request.body()
        headers = dict(request.headers)
        results = await self._broadcast_to_workers("update_weights_from_disk", body, headers)
        return JSONResponse(content={"results": results})

    async def add_worker(self, request: Request):
        """Register a new diffusion worker."""
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        if not worker_url:
            body = await request.body()
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            if self.verbose:
                print(f"[diffusion-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": list(self.worker_request_counts.keys())}

    async def list_workers(self, request: Request):
        """List all registered workers."""
        return {"urls": list(self.worker_request_counts.keys())}

    async def proxy(self, request: Request, path: str):
        """Catch-all: forward any unmatched request to the least-loaded worker."""
        return await self._forward_to_worker(request, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Miles Diffusion Router")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30080)
    parser.add_argument("--worker-urls", nargs="*", default=[], help="Initial worker URLs to register")
    parser.add_argument("--max-connections", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--health-check-interval", type=int, default=10)
    parser.add_argument("--health-check-failure-threshold", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    router = DiffusionRouter(args, verbose=args.verbose)
    for url in args.worker_urls:
        router.worker_request_counts[url] = 0
        router.worker_failure_counts[url] = 0

    uvicorn.run(router.app, host=args.host, port=args.port, log_level="info")
