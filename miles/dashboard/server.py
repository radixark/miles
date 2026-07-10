"""HTTP API of the miles dashboard.

``make_app(store, reader)`` wires the two read-side data sources — the
:class:`MetricStore` (JSONL telemetry under ``{dump_details}/dashboard/``)
and the :class:`DumpReader` (rollout/train ``.pt`` dumps) — into the REST
API consumed by the SPA. The server is strictly read-only over files on
disk; live viewing is the same app with a follow loop tailing the store
(see ``serve.py``).

Metric catalog: keys from ``metrics.jsonl`` are served as-is; dump-derived
per-step aggregates are namespaced as ``dump/<column>`` so the L0 view works
even for runs where the dashboard backend was not enabled during training.

Error mapping: ``DumpStillWriting`` -> 503 (client retries),
``FileNotFoundError``/``KeyError`` -> 404, ``ValueError`` -> 400.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from miles.dashboard.dump_reader import STEP_AGGREGATE_METRICS, DumpReader, DumpStillWriting
from miles.dashboard.store import MetricStore, Stream

DUMP_METRIC_PREFIX = "dump/"

_STATIC_DIR = Path(__file__).parent / "static"


@contextmanager
def _translate_errors():
    try:
        yield
    except DumpStillWriting as e:
        raise HTTPException(status_code=503, detail=f"dump still being written: {e}") from e
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _json_safe(value):
    """NaN/inf are not valid JSON; per-token GPU outputs can contain them."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _table(df) -> dict:
    return dict(columns=df.columns, rows=[_json_safe(row) for row in df.to_dicts()])


def make_app(store: MetricStore, reader: DumpReader, *, follow: bool = False) -> FastAPI:
    app = FastAPI(title="miles dashboard", docs_url=None, redoc_url=None)

    @app.get("/api/meta")
    def meta():
        ids = reader.rollout_ids()
        # dump-derived aggregates are the L0 fallback for dump-only dirs; a
        # run with a telemetry stream never advertises them (they torch.load
        # raw sample dumps — the /api/metrics handler still serves explicit
        # requests, but the catalog stays wandb-shaped)
        offline = ids.train and not store.records[Stream.METRICS]
        dump_keys = [DUMP_METRIC_PREFIX + column for column in STEP_AGGREGATE_METRICS] if offline else []
        return dict(
            mode="follow" if follow else "static",
            run_name=store.meta.run_name if store.meta else None,
            start_ts=store.meta.start_ts if store.meta else None,
            time_range=store.time_range(),
            rollout_ids=dict(train=ids.train, eval=ids.eval),
            metric_keys=store.metric_keys() + dump_keys,
            engine_metric_keys=store.engine_metric_names(),
            step_keys=store.step_keys(),
            capabilities=dict(
                has_metrics=store.has_stream(Stream.METRICS),
                has_tokenizer=(reader.dump_dir / "tokenizer").is_dir(),
                has_timeline=store.has_stream(Stream.PHASES) or store.has_stream(Stream.GPU_UTIL),
                has_engine_series=store.has_stream(Stream.ENGINE_SERIES),
                max_window_s=MetricStore.MAX_WINDOW_S,
            ),
        )

    def _check_window(t0: float | None, t1: float | None) -> None:
        if t0 is None or t1 is None:
            return
        if t1 <= t0:
            raise ValueError(f"bad window: {t1=} <= {t0=}")
        if t1 - t0 > MetricStore.MAX_WINDOW_S:
            raise ValueError(f"window {t1 - t0:.0f}s exceeds max_window_s {MetricStore.MAX_WINDOW_S:.0f}")

    # ------------------------------ timeline --------------------------------

    @app.get("/api/timeline/topology")
    def timeline_topology():
        return dict(lanes=store.lanes(), windows=store.topology_windows())

    @app.get("/api/timeline/phases")
    def timeline_phases(t0: float | None = None, t1: float | None = None, lanes: str | None = None):
        with _translate_errors():
            _check_window(t0, t1)
            return dict(phases=store.phases_by_lane(t0=t0, t1=t1, lanes=store.resolve_lanes(lanes)))

    @app.get("/api/timeline/gpu")
    def timeline_gpu(
        t0: float | None = None, t1: float | None = None, max_points: int = 2000, lanes: str | None = None
    ):
        with _translate_errors():
            _check_window(t0, t1)
            if max_points < 2:
                raise ValueError(f"{max_points=} must be >= 2")
            return dict(lanes=store.gpu_series(t0=t0, t1=t1, max_points=max_points, lanes=store.resolve_lanes(lanes)))

    @app.get("/api/timeline/heatmap")
    def timeline_heatmap(
        metric: str = "util",
        t0: float | None = None,
        t1: float | None = None,
        x_buckets: int = 1200,
        lanes: str | None = None,
    ):
        """Binary rank carpet: [4-byte LE header length][header JSON][uint8
        matrix, row-major] — one byte per (lane, time bucket) cell."""
        with _translate_errors():
            _check_window(t0, t1)
            if not 2 <= x_buckets <= 4000:
                raise ValueError(f"{x_buckets=} out of range [2, 4000]")
            result = store.heatmap(metric, t0=t0, t1=t1, x_buckets=x_buckets, lanes=store.resolve_lanes(lanes))
            values = result.pop("values")
            header = json.dumps(_json_safe(result)).encode()
            return Response(
                content=len(header).to_bytes(4, "little") + header + values,
                media_type="application/octet-stream",
            )

    @app.get("/api/timeline/outliers")
    def timeline_outliers(criterion: str, t0: float | None = None, t1: float | None = None, top_k: int = 16):
        with _translate_errors():
            if not 1 <= top_k <= 256:
                raise ValueError(f"{top_k=} out of range [1, 256]")
            return dict(outliers=store.outliers(criterion, t0=t0, t1=t1, top_k=top_k))

    @app.get("/api/timeline/engine_series")
    def timeline_engine_series(metric: str, t0: float | None = None, t1: float | None = None, max_points: int = 2000):
        with _translate_errors():
            _check_window(t0, t1)
            if max_points < 2:
                raise ValueError(f"{max_points=} must be >= 2")
            return dict(series=store.engine_series(metric, t0=t0, t1=t1, max_points=max_points))

    @app.get("/api/timeline/bubbles")
    def timeline_bubbles():
        return dict(bubbles=store.bubbles())

    @app.get("/api/metrics")
    def metrics(keys: str, x: str = "rollout/step", t0: float | None = None, t1: float | None = None):
        with _translate_errors():
            requested = [k for k in keys.split(",") if k]
            if not requested:
                raise ValueError("keys must be a non-empty comma-separated list")
            store_keys = [k for k in requested if not k.startswith(DUMP_METRIC_PREFIX)]
            series = store.metric_series(store_keys, x_key=x, t0=t0, t1=t1)
            dump_keys = [k for k in requested if k.startswith(DUMP_METRIC_PREFIX)]
            if dump_keys:
                aggregates = reader.step_aggregates()
                steps = aggregates["rollout_id"].to_list() if aggregates.height else []
                for key in dump_keys:
                    column = key.removeprefix(DUMP_METRIC_PREFIX)
                    if column not in STEP_AGGREGATE_METRICS:
                        raise ValueError(f"unknown dump metric {key!r}")
                    points = [
                        (step, value)
                        for step, value in zip(steps, aggregates[column].to_list(), strict=True)
                        if value is not None
                    ]
                    series[key] = dict(x=[p[0] for p in points], y=[_json_safe(p[1]) for p in points], ts=[])
            return _json_safe(series)

    @app.get("/api/rollout/{rollout_id}/summary")
    def rollout_summary(rollout_id: int, evaluation: bool = Query(False, alias="eval")):
        with _translate_errors():
            df = reader.summary(rollout_id, evaluation=evaluation)
            return dict(rollout_id=rollout_id, evaluation=evaluation, **_table(df))

    @app.get("/api/rollout/{rollout_id}/groups")
    def rollout_groups(rollout_id: int, evaluation: bool = Query(False, alias="eval")):
        with _translate_errors():
            df = reader.groups(rollout_id, evaluation=evaluation)
            return dict(rollout_id=rollout_id, evaluation=evaluation, **_table(df))

    @app.get("/api/rollout/{rollout_id}/sample/{sample_index}/tokens")
    def sample_tokens(
        rollout_id: int,
        sample_index: int,
        start: int = 0,
        end: int | None = None,
        evaluation: bool = Query(False, alias="eval"),
    ):
        with _translate_errors():
            payload = reader.tokens(rollout_id, sample_index, start=start, end=end, evaluation=evaluation)
            return _json_safe(payload)

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.middleware("http")
        async def _no_stale_frontend(request, call_next):
            # without Cache-Control browsers cache heuristically and keep
            # serving a STALE SPA after a stack upgrade; no-cache forces a
            # revalidation (cheap 304s) so the frontend always matches serve
            response = await call_next(request)
            if request.url.path == "/" or request.url.path.startswith("/static"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(_STATIC_DIR / "index.html")

    return app
