"""
Demo script for the Miles Diffusion Router.

Starts a diffusion router that load-balances requests across multiple
sglang-diffusion worker instances using least-request routing.

Usage:
    # Start with no workers (add them dynamically via /add_worker):
    python examples/diffusion_router/demo.py --port 30080

    # Start with pre-registered workers:
    python examples/diffusion_router/demo.py --port 30080 \
        --worker-urls http://localhost:10090 http://localhost:10091

    # Then interact:
    curl http://localhost:30080/health
    curl -X POST 'http://localhost:30080/add_worker?url=http://localhost:10092'
    curl http://localhost:30080/list_workers
    curl -X POST http://localhost:30080/generate -H 'Content-Type: application/json' \
        -d '{"model": "stabilityai/stable-diffusion-3", "prompt": "a cat", "n": 1, "size": "1024x1024"}'
"""

import argparse

import uvicorn

from miles.router.diffusion_router import DiffusionRouter


def main():
    parser = argparse.ArgumentParser(description="Miles Diffusion Router Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Router bind address")
    parser.add_argument("--port", type=int, default=30080, help="Router port")
    parser.add_argument("--worker-urls", nargs="*", default=[], help="Initial diffusion worker URLs")
    parser.add_argument("--max-connections", type=int, default=100, help="Max concurrent connections to workers")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds")
    parser.add_argument("--health-check-interval", type=int, default=10, help="Seconds between health checks")
    parser.add_argument("--health-check-failure-threshold", type=int, default=3, help="Failures before quarantine")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    router = DiffusionRouter(args, verbose=args.verbose)

    # Pre-register any workers specified on the command line
    for url in args.worker_urls:
        router.worker_request_counts[url] = 0
        router.worker_failure_counts[url] = 0
        if args.verbose:
            print(f"[demo] Pre-registered worker: {url}")

    print(f"[demo] Starting diffusion router on {args.host}:{args.port}")
    print(f"[demo] Workers: {list(router.worker_request_counts.keys()) or '(none — add via POST /add_worker)'}")
    uvicorn.run(router.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
