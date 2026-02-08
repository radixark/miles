# Miles Diffusion Router

Load-balances requests across multiple `sglang-diffusion` worker instances using least-request routing with background health checks and worker quarantine.

## Quick Start

```bash
# Start the router with two diffusion backends
python examples/diffusion_router/demo.py --port 30080 \
    --worker-urls http://localhost:10090 http://localhost:10091

# Or start empty and add workers dynamically
python examples/diffusion_router/demo.py --port 30080
curl -X POST 'http://localhost:30080/add_worker?url=http://localhost:10090'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Image generation (forwards to `/v1/images/generations`) |
| `POST` | `/generate_video` | Video generation (forwards to `/v1/videos`) |
| `GET` | `/health` | Aggregated router health |
| `GET` | `/health_workers` | Per-worker health and load info |
| `POST` | `/add_worker` | Register a diffusion worker (`?url=...` or JSON body) |
| `GET` | `/list_workers` | List registered workers |
| `POST` | `/update_weights_from_disk` | Broadcast weight reload to all workers |
| `GET, POST, PUT, DELETE` | `/{path}` | Catch-all proxy to least-loaded worker |

## Load Balancing

The router uses a **least-request** strategy: each incoming request is forwarded to the worker with the fewest in-flight requests. This is workload-aware and avoids hot-spotting compared to round-robin. When a request completes, the worker's count is decremented, keeping the load state accurate in real time.

Workers that fail consecutive health checks (default: 3) are quarantined and excluded from the routing pool and will not be added back to avoid stale weights(See [discussion](https://github.com/radixark/miles/pull/260#discussion_r2654274449)). A background loop pings each worker's `GET /health` endpoint at a configurable interval (default: 10s).

## Benchmark script(under examples/diffusion_router/):
- examples/diffusion_router/bench_routing_algorithms.py — top-level comparison runner
- examples/diffusion_router/bench_router.py — single-algorithm benchmark (spawned per algorithm)
- examples/diffusion_router/demo.py — router process (spawned by bench_router.py)

## Notes

- Health check endpoint follows Miles/SGLang convention: `GET /health`.
- Responses are fully buffered; streaming and large-response handling are not supported yet (planned for a follow-up PR).

## Example Requests

```bash
# Check health
curl http://localhost:30080/health

# Generate an image
curl -X POST http://localhost:30080/generate \
    -H 'Content-Type: application/json' \
    -d '{"model": "stabilityai/stable-diffusion-3", "prompt": "a cat", "n": 1, "size": "1024x1024"}'

# Reload weights on all workers
curl -X POST http://localhost:30080/update_weights_from_disk \
    -H 'Content-Type: application/json' \
    -d '{"model_path": "/path/to/new/weights"}'
```

## CLI Options

```
--host                          Bind address (default: 0.0.0.0)
--port                          Port (default: 30080)
--worker-urls                   Initial worker URLs
--max-connections               Max concurrent connections (default: 100)
--timeout                       Request timeout in seconds for router-to-worker requests
--health-check-interval         Seconds between health checks (default: 10)
--health-check-failure-threshold  Failures before quarantine (default: 3)
--verbose                       Enable verbose logging
```
