"""Ray-actor wrapper around a ``dynamo.sglang`` worker subprocess.

Drop-in replacement for :class:`miles.backends.sglang_utils.sglang_engine.SGLangEngine`
when ``--rollout-backend dynamo`` is active. The worker process is still
SGLang under the hood; ``dynamo.sglang`` only adds:

* Discovery: the worker self-registers via etcd (or the file-backed
  discovery shim) so Dynamo's frontend can find it.
* KV events: each worker can publish per-block cache events over ZMQ, which
  the Dynamo frontend subscribes to (via NATS) when ``--router-mode kv`` is
  selected. Without this stream the router falls back to its approximate
  ``predict-on-route`` policy.
* The ``/engine/*`` control-plane routes (weight update, flush, pause, …)
  that Aphoh's ``RLMixin`` exposes on top of SGLang's native HTTP server.

Lifecycle: Miles owns the worker via a Ray actor (this class).  We start
the subprocess in :meth:`init` and tear it down in :meth:`shutdown`.  The
worker's HTTP port serves both SGLang's native ``/generate`` etc. *and*
Dynamo's ``/engine/*`` routes — we can therefore reuse Miles' existing
``_make_request`` helper pattern.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from typing import Optional

import requests
from sglang.srt.utils import kill_process_tree

from miles.backends.sglang_utils.sglang_engine import (
    SGLangEngine,
    _compute_server_args,
    _wait_server_healthy,
)
from miles.ray.ray_actor import RayActor
from miles.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment toggles. Keep them as module-level constants so call sites
# (and tests) can patch them without poking into the launching subprocess.
# ---------------------------------------------------------------------------
_ENV_DISCOVERY_DEFAULT = "etcd"  # `file` for the no-etcd local-dev mode
_ENV_KV_EVENTS_PORT_OFFSET = 10000  # server_port + offset → ZMQ KV-events port


def _bracket_v6(addr: Optional[str]) -> Optional[str]:
    """Wrap a bare IPv6 address in ``[...]`` for use in HTTP URLs."""
    if not addr or addr.startswith("["):
        return addr
    try:
        socket.inet_pton(socket.AF_INET6, addr)
        return f"[{addr}]"
    except OSError:
        return addr


class DynamoEngine(RayActor):
    """Owns one ``python -m dynamo.sglang`` worker.

    Public method surface mirrors :class:`SGLangEngine` so that
    :class:`miles.ray.rollout.ServerGroup` can wrap it with
    ``ray.remote(DynamoEngine)`` and the rest of Miles continues unchanged.
    """

    # ------------------------------------------------------------------
    # Construction (state only — no IO)
    # ------------------------------------------------------------------
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine

        # Populated by init().
        self.process: Optional[subprocess.Popen] = None
        self.node_rank: int = 0
        self.server_host: Optional[str] = None
        self.server_port: Optional[int] = None
        self.router_ip: Optional[str] = None
        self.router_port: Optional[int] = None

    # ------------------------------------------------------------------
    # init() — same signature as SGLangEngine.init
    # ------------------------------------------------------------------
    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
        engine_info_bootstrap_port=None,
    ):
        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port

        host = _bracket_v6(host or get_host_info()[1])
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_bracket_v6(ip_part)}:{port_part}"

        # Reuse SGLang's server-arg derivation — TP/PP topology, GPU IDs,
        # node-rank assignment etc. are identical.
        server_args, _ = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            engine_info_bootstrap_port=engine_info_bootstrap_port,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )
        self.node_rank = server_args["node_rank"]
        self.server_host = server_args["host"]
        self.server_port = server_args["port"]

        if self.node_rank != 0:
            # Multi-node workers: only rank-0 owns the process; other nodes
            # piggy-back via SGLang's nnodes/node_rank cluster mode.
            return

        argv = self._build_argv(server_args)
        env = self._build_env(server_args)

        logger.info("DynamoEngine[rank=%d] launching: %s", self.rank, " ".join(argv))
        self.process = subprocess.Popen(argv, env=env)

        # ``dynamo.sglang``'s system-status server exposes /health (not
        # /health_generate which is SGLang-native). Use that for readiness.
        self._wait_for_dynamo_health(timeout_s=600)

    def _wait_for_dynamo_health(self, timeout_s: int = 600) -> None:
        url = f"http://{self.server_host}:{self.server_port}/health"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200 and "ready" in resp.text:
                    logger.info("DynamoEngine[rank=%d] healthy at %s", self.rank, url)
                    return
            except requests.RequestException:
                pass
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"DynamoEngine[rank={self.rank}] subprocess exited "
                    f"with code {self.process.returncode}"
                )
            time.sleep(2)
        raise TimeoutError(
            f"DynamoEngine[rank={self.rank}] did not become healthy at {url} "
            f"within {timeout_s}s"
        )

    # ------------------------------------------------------------------
    # Subprocess command + env construction
    # ------------------------------------------------------------------
    def _build_argv(self, server_args: dict) -> list[str]:
        """Translate Miles' SGLang server-args dict into a ``dynamo.sglang`` CLI."""
        args = self.args
        argv: list[str] = [sys.executable, "-m", "dynamo.sglang"]

        # Core SGLang flags that ``dynamo.sglang`` understands directly.
        argv += ["--model-path", server_args["model_path"]]
        argv += ["--host", str(server_args["host"]).strip("[]")]
        argv += ["--port", str(server_args["port"])]
        argv += ["--tp", str(server_args["tp_size"])]
        if server_args.get("pp_size", 1) > 1:
            argv += ["--pp-size", str(server_args["pp_size"])]
        if server_args.get("dp_size", 1) > 1:
            argv += ["--dp-size", str(server_args["dp_size"])]
        if server_args.get("trust_remote_code"):
            argv += ["--trust-remote-code"]
        if server_args.get("mem_fraction_static") is not None:
            argv += ["--mem-fraction-static", str(server_args["mem_fraction_static"])]
        if server_args.get("dtype"):
            argv += ["--dtype", str(server_args["dtype"])]

        # Multi-node SGLang cluster knobs (passed through transparently).
        for key in ("nnodes", "node_rank", "dist_init_addr"):
            if server_args.get(key) is not None:
                argv += [f"--{key.replace('_', '-')}", str(server_args[key])]

        # PD-disagg (matches SGLangEngine.init_normal's logic).
        if self.worker_type in ("prefill", "decode"):
            argv += ["--disaggregation-mode", self.worker_type]
            bp = server_args.get("disaggregation_bootstrap_port")
            if bp:
                argv += ["--disaggregation-bootstrap-port", str(bp)]

        # ----- Dynamo-specific -----
        argv += ["--enable-rl"]

        # Page size > 1 is mandatory for KV routing (the router builds a
        # block-hash prefix tree; ``page_size=1`` makes the router panic
        # because per-token blocks defeat the prefix-grouping).
        page_size = getattr(args, "dynamo_page_size", 64)
        argv += ["--page-size", str(page_size)]

        # If KV routing is requested, point the worker at a ZMQ port for
        # publishing block-allocation events; Dynamo's frontend bridges
        # these to NATS.
        if self._kv_router_enabled():
            zmq_port = self.server_port + _ENV_KV_EVENTS_PORT_OFFSET
            kv_cfg = json.dumps(
                {
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "endpoint": f"tcp://*:{zmq_port}",
                    "enable_kv_cache_events": True,
                }
            )
            argv += ["--kv-events-config", kv_cfg]

        # Per-group overrides from ServerGroupConfig.sglang_overrides
        # (mirrors what SGLangEngine does, minus the handful of keys we set
        # explicitly above).
        already = {"model_path", "host", "port", "tp", "tp_size", "pp", "pp_size"}
        for k, v in self.sglang_overrides.items():
            if k in already:
                continue
            argv += [f"--{k.replace('_', '-')}", str(v)]

        return argv

    def _build_env(self, server_args: dict) -> dict:
        env = os.environ.copy()
        env["DYN_DISCOVERY_BACKEND"] = self._discovery_backend()

        # The Dynamo system-status server (which exposes /health and
        # /engine/* routes) only binds when this is set.  We co-locate it on
        # the SGLang HTTP port so Miles can reuse one URL for both planes.
        env["DYN_SYSTEM_PORT"] = str(self.server_port)

        # KV block hashes must be deterministic across workers — otherwise
        # the frontend's predicted-block hashes won't match what workers
        # actually publish, and cache hits collapse to 0%.
        env.setdefault("PYTHONHASHSEED", "0")

        # ``dynamo.sglang``'s decoded-token stream adds detokenizer-side
        # latency. Miles always aggregates the full response, so disable
        # streaming to recover the ~5% throughput hit.
        env.setdefault("DYN_SGL_FORCE_NONSTREAM", "1")

        # Pin the subprocess to its assigned GPU slice. Miles' ServerGroup
        # creates each actor with NOSET_VISIBLE_DEVICES set (it manages GPU
        # placement via base_gpu_id explicitly), which means the actor's
        # CUDA_VISIBLE_DEVICES is the host's full set (0..N-1). Without an
        # override here every engine subprocess would default to GPUs 0..tp-1
        # — they'd all stack on the same physical pair and OOM.
        base = int(server_args.get("base_gpu_id", 0))
        span = int(server_args["tp_size"]) * int(server_args.get("pp_size", 1) or 1)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(base + i) for i in range(span))
        return env

    def _discovery_backend(self) -> str:
        # Etcd is required for KV routing; the file-backed shim is fine for
        # the simpler round-robin / random modes.
        if self._kv_router_enabled():
            return "etcd"
        return getattr(self.args, "dynamo_discovery_backend", "file")

    def _kv_router_enabled(self) -> bool:
        return getattr(self.args, "dynamo_router_mode", "round-robin") == "kv"

    # ------------------------------------------------------------------
    # HTTP plumbing — proxy to the same SGLang endpoints SGLangEngine uses.
    # ``dynamo.sglang`` keeps SGLang's native /flush_cache, /pause_generation
    # etc. routes, so the bodies below are intentionally the same as
    # SGLangEngine's. The win is that we don't have to re-derive any of
    # SGLang's logic.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Two control-plane HTTP helpers, matching what dynamo.sglang's RLMixin
    # exposes on each worker's system-status server.
    #
    #   _call_engine_route("name", body)
    #     POST /engine/{name}   body is the request payload directly
    #     Used for weight-transfer paths that Dynamo modeled as first-class
    #     engine routes (update_weights_from_distributed, ..._from_tensor,
    #     ..._from_disk).
    #
    #   _call_tokenizer_manager("method", args, kwargs)
    #     POST /engine/call_tokenizer_manager  (generic passthrough from
    #     ai-dynamo/dynamo PR #6836)
    #     Body shape: {"method": "<name>", "args": [...], "kwargs": {...}}
    #     Used for everything else (flush_cache, pause/continue_generation,
    #     init/destroy_weights_update_group, post_process_weights, ...).
    #     Most TM methods expect their args wrapped as
    #     [{"io_struct.<Type>ReqInput": {...}}] — matches sglang's req model.
    # ------------------------------------------------------------------
    def _call_engine_route(self, route: str, body: dict | None = None):
        if self.node_rank != 0:
            return None
        url = f"http://{self.server_host}:{self.server_port}/engine/{route}"
        timeout = float(os.getenv("MILES_DYNAMO_ENGINE_ROUTE_TIMEOUT", "1800"))
        resp = requests.post(url, json=body or {}, timeout=timeout)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{resp.text=}")
            raise
        return resp.json()

    def _call_tokenizer_manager(self, method: str, args=None, kwargs=None):
        if self.node_rank != 0:
            return None
        body: dict = {"method": method}
        if args is not None:
            body["args"] = args
        if kwargs is not None:
            body["kwargs"] = kwargs
        return self._call_engine_route("call_tokenizer_manager", body)

    # Backwards-compat shim: SGLangEngine code that uses ``_make_request``
    # still works for the routes we map onto _call_engine_route.
    def _make_request(self, endpoint: str, payload: dict | None = None):
        return self._call_engine_route(endpoint, payload)

    # ------------------------------------------------------------------
    # Weight-transfer routes use the dedicated /engine/<name> endpoints.
    # ------------------------------------------------------------------
    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        body = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            body["weight_version"] = weight_version
        return self._call_engine_route("update_weights_from_tensor", body)

    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        body = {
            "names": names,
            "dtypes": [str(d).replace("torch.", "") for d in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            body["weight_version"] = weight_version
        return self._call_engine_route("update_weights_from_distributed", body)

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        body: dict = {"model_path": model_path}
        if load_format is not None:
            body["load_format"] = load_format
        return self._call_engine_route("update_weights_from_disk", body)

    # ------------------------------------------------------------------
    # Everything else goes through call_tokenizer_manager.
    # Args are wrapped in sglang's io_struct request shape.
    # ------------------------------------------------------------------
    def init_weights_update_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend,
    ):
        return self._call_tokenizer_manager(
            "init_weights_update_group",
            args=[{
                "io_struct.InitWeightsUpdateGroupReqInput": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": backend,
                },
            }],
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._call_tokenizer_manager(
                "destroy_weights_update_group",
                args=[{
                    "io_struct.DestroyWeightsUpdateGroupReqInput": {
                        "group_name": group_name,
                    },
                }],
            )
        except requests.RequestException:
            pass

    def flush_cache(self):
        if self.node_rank != 0:
            return
        self._call_tokenizer_manager("flush_cache")

    def abort_request(self, abort_all: bool = True):
        """Cancel in-flight requests on this worker.

        Goes through the same ``/engine/call_tokenizer_manager`` passthrough
        every other non-weight-transfer control op uses — so the entire
        control plane is uniform: weight transfer through dedicated
        ``/engine/<route>`` endpoints, everything else through the
        tokenizer-manager dispatcher. Avoids assuming
        ``DYN_SYSTEM_PORT == SGLang native HTTP port``.
        """
        if self.node_rank != 0:
            return
        try:
            return self._call_tokenizer_manager(
                "abort_request",
                args=[{"io_struct.AbortReq": {"abort_all": abort_all}}],
            )
        except Exception as e:
            logger.warning("DynamoEngine[rank=%d] abort_request failed: %s", self.rank, e)

    def pause_generation(self, mode: str = "retract"):
        if self.node_rank != 0:
            return
        return self._call_tokenizer_manager(
            "pause_generation",
            args=[{"io_struct.PauseGenerationReqInput": {}}],
        )

    def continue_generation(self):
        if self.node_rank != 0:
            return
        return self._call_tokenizer_manager(
            "continue_generation",
            args=[{"io_struct.ContinueGenerationReqInput": {}}],
        )

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
        post_load_weights: bool = False,
    ):
        return self._call_tokenizer_manager(
            "post_process_weights",
            args=[{
                "io_struct.PostProcessWeightsReqInput": {
                    "restore_weights_before_load": restore_weights_before_load,
                    "post_process_quantization": post_process_quantization,
                    "post_load_weights": post_load_weights,
                },
            }],
        )

    def check_weights(self, action: str):
        return self._call_tokenizer_manager(
            "check_weights",
            args=[{"io_struct.CheckWeightsReqInput": {"action": action}}],
        )

    # release/resume memory occupation on dynamo workers also unregisters
    # from discovery, which would break the frontend's routing. Slynamo
    # currently no-ops these — match that behavior; we just flush the cache.
    def release_memory_occupation(self, tags: list[str] | None = None):
        self.flush_cache()

    def resume_memory_occupation(self, tags: list[str] | None = None):
        pass

    def get_weight_version(self):
        # Not exposed via call_tokenizer_manager in current dynamo build.
        return None

    update_weight_version = SGLangEngine.update_weight_version
    start_profile = SGLangEngine.start_profile
    stop_profile = SGLangEngine.stop_profile

    # ------------------------------------------------------------------
    # Health & info — read from Dynamo's /health (NOT /health_generate).
    # ------------------------------------------------------------------
    def health_generate(self, timeout: float = 5.0) -> bool:
        if self.node_rank != 0:
            return True
        resp = requests.get(
            f"http://{self.server_host}:{self.server_port}/health",
            timeout=timeout,
        )
        resp.raise_for_status()
        return '"status":"ready"' in resp.text

    def get_server_info(self):
        resp = requests.get(
            f"http://{self.server_host}:{self.server_port}/health",
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------
    def shutdown(self):
        if self.process is None:
            return
        logger.info("DynamoEngine[rank=%d] shutting down (pid=%d)", self.rank, self.process.pid)
        kill_process_tree(self.process.pid)
        self.process = None

    def simulate_crash(self):
        # Used by Miles' fault-tolerance tests. Same semantics: hard-kill the
        # subprocess; supervisor (Ray actor) should observe the death.
        self.shutdown()
