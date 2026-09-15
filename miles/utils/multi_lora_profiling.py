import json
import logging
import statistics
import time
from functools import wraps

logger = logging.getLogger(__name__)

PROFILE_LOG_PREFIX = "multi-LoRA profile: "
METRICS_LOG_PREFIX = "multi-LoRA metrics: "
REQUESTS_LOG_PREFIX = "multi-LoRA requests: "

TRAINER_OPS = (
    "load_slot",
    "unload_slot",
    "forward_backward",
    "forward_only",
    "optim_step",
    "save_slot",
    "export_slot",
)
ENGINE_OPS = ("sample",)
API_PATHS = {
    "/api/v1/create_model": "create_model",
    "/api/v1/forward_backward": "forward_backward",
    "/api/v1/optim_step": "optim_step",
    "/api/v1/save_weights": "save_weights",
    "/api/v1/load_weights": "load_weights",
    "/api/v1/save_weights_for_sampler": "save_weights_for_sampler",
    "/api/v1/asample": "asample",
}
RETRIEVE_PATH = "/api/v1/retrieve_future"


def _quantile(values: list[float], fraction: float) -> float:
    return values[min(len(values) - 1, int(fraction * len(values)))]


class Profiler:
    def __init__(self) -> None:
        self._seconds: dict[str, list[float]] = {}
        self._sizes: dict[str, int] = {}
        self._series: dict[str, dict[str, list[float]]] = {}
        self._requests: list[list] = []

    def record(self, op: str, seconds: float, size: int = 1) -> None:
        self._seconds.setdefault(op, []).append(seconds)
        self._sizes[op] = self._sizes.get(op, 0) + size

    def observe(self, slot: int, **metrics: float) -> None:
        for name, value in metrics.items():
            self._series.setdefault(str(slot), {}).setdefault(name, []).append(float(value))

    def request(self, op: str, tenant: str, created_at: float, finished_at: float) -> None:
        self._requests.append([tenant, op, round(created_at, 3), round(finished_at, 3)])

    def snapshot(self) -> dict[str, dict[str, float]]:
        trainer_total = sum(sum(values) for op, values in self._seconds.items() if op in TRAINER_OPS) or 1.0
        snapshot = {}
        for op, values in self._seconds.items():
            ordered = sorted(values)
            snapshot[op] = {
                "calls": len(values),
                "size": self._sizes[op],
                "total_s": sum(values),
                "mean_s": statistics.fmean(values),
                "p50_s": _quantile(ordered, 0.5),
                "p90_s": _quantile(ordered, 0.9),
                "max_s": ordered[-1],
                "share": sum(values) / trainer_total if op in TRAINER_OPS else 0.0,
            }
        return snapshot

    def series(self) -> dict[str, dict[str, list[float]]]:
        return self._series

    def drain_requests(self) -> list[list]:
        requests, self._requests = self._requests, []
        return requests

    def log(self) -> None:
        if self._seconds:
            logger.info("%s%s", PROFILE_LOG_PREFIX, json.dumps(self.snapshot()))
        if self._series:
            logger.info("%s%s", METRICS_LOG_PREFIX, json.dumps(self._series))
        if requests := self.drain_requests():
            logger.info("%s%s", REQUESTS_LOG_PREFIX, json.dumps(requests))


def _call_size(op: str, args: tuple, kwargs: dict) -> int:
    if op in ("forward_backward", "forward_only"):
        return len(args[1] if len(args) > 1 else kwargs["slot_datums"])
    if op == "optim_step":
        return len(args[0] if args else kwargs["adam_params_by_slot"])
    if op == "sample":
        return (args[0] if args else kwargs["payload"]).get("num_samples", 1)
    return 1


def _response_logprobs(datum: dict, output: dict) -> list[float]:
    mask = datum.get("weights") or datum.get("advantages")
    if mask is None:
        return list(output["logprobs"])
    return [value for value, keep in zip(output["logprobs"], mask, strict=False) if keep]


def _observe_step(profiler: Profiler, slot_datums: list, outputs) -> None:
    if not isinstance(outputs, list):
        return
    per_slot: dict[int, list[tuple[dict, dict]]] = {}
    for (slot, datum), output in zip(slot_datums, outputs, strict=True):
        per_slot.setdefault(slot, []).append((datum, output))
    for slot, pairs in per_slot.items():
        logprobs = [value for datum, output in pairs for value in _response_logprobs(datum, output)]
        profiler.observe(
            slot,
            loss=sum(output["loss"] for _, output in pairs) / len(pairs),
            log_prob=sum(logprobs) / len(logprobs) if logprobs else 0.0,
            mean_len=len(logprobs) / len(pairs),
        )


def _timed(profiler: Profiler, op: str, method):
    @wraps(method)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await method(*args, **kwargs)
        finally:
            profiler.record(op, time.perf_counter() - start, _call_size(op, args, kwargs))
        if op == "forward_backward":
            _observe_step(profiler, args[1] if len(args) > 1 else kwargs["slot_datums"], result)
        if op in ("forward_backward", "optim_step"):
            profiler.log()
        return result

    return wrapper


def instrument_backend(backend):
    profiler = Profiler()
    for op in TRAINER_OPS + ENGINE_OPS:
        if (method := getattr(backend, op, None)) is not None:
            setattr(backend, op, _timed(profiler, op, method))
    backend.profiler = profiler
    return backend


class _RequestTiming:
    def __init__(self, app, profiler: Profiler) -> None:
        self.app = app
        self.profiler = profiler
        self.pending: dict[str, tuple[str, str, float]] = {}

    async def __call__(self, scope, receive, send):
        path = scope.get("path")
        if scope["type"] != "http" or (path not in API_PATHS and path != RETRIEVE_PATH):
            await self.app(scope, receive, send)
            return
        arrival = time.monotonic()
        messages = []
        while True:
            messages.append(await receive())
            if messages[-1]["type"] != "http.request" or not messages[-1].get("more_body"):
                break
        request_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.request")
        response_chunks: list[bytes] = []

        async def replay():
            return messages.pop(0) if messages else await receive()

        async def send_recorded(message):
            if message["type"] == "http.response.body":
                response_chunks.append(message.get("body", b""))
            await send(message)

        await self.app(scope, replay, send_recorded)
        tenant = next((value.decode() for key, value in scope["headers"] if key == b"x-api-key"), "")
        self._settle(path, tenant, request_body, b"".join(response_chunks), arrival)

    def _settle(self, path: str, tenant: str, request_body: bytes, response_body: bytes, arrival: float) -> None:
        if path in API_PATHS:
            if (request_id := _json_field(response_body, "request_id")) is not None:
                self.pending[request_id] = (API_PATHS[path], tenant, arrival)
            return
        request_id = _json_field(request_body, "request_id")
        if request_id not in self.pending or _json_field(response_body, "type") == "try_again":
            return
        op, tenant, created_at = self.pending.pop(request_id)
        if _json_field(response_body, "error") is None:
            self.profiler.request(op, tenant, created_at, time.monotonic())


def _json_field(body: bytes, key: str):
    try:
        value = json.loads(body)
    except ValueError:
        return None
    return value.get(key) if isinstance(value, dict) else None


def instrument_app(app, profiler: Profiler):
    app.add_middleware(_RequestTiming, profiler=profiler)
    return app
