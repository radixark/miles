"""Launch a Dynamo frontend in place of ``sgl_router``.

The HTTP surface the rest of Miles sees is *almost* the same — it still
POSTs to ``http://{router_ip}:{router_port}/...`` and reads the response
back — except the path is ``/v1/completions`` (OpenAI-style) instead of
``/generate`` (SGLang native).  That dispatch happens inside the rollout
function; see :func:`miles.rollout.sglang_rollout.generate`.

This module just owns the *startup* side:

* If ``--dynamo-router-mode kv`` is requested, the frontend needs etcd + a
  NATS broker for the discovery and KV-events transports respectively. We
  launch both as subprocesses on the box if they aren't already running —
  exactly the same pattern we use for SGLang's router today.
* Then we launch ``python -m dynamo.frontend`` and block on its ``/health``.

Returned ``(ip, port)`` slots straight into ``args.sglang_router_ip`` and
``args.sglang_router_port`` so the rest of Miles keeps working.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import requests

from miles.utils.http_utils import (
    _wrap_ipv6,
    find_available_port,
    get_host_info,
    is_port_available,
    wait_for_server_ready,
)

logger = logging.getLogger(__name__)


# Module-level cache so repeated `_start_router(force_new=True)` calls don't
# stand up a second copy of etcd / nats. Each side-car is keyed by its port
# so we can detect a pre-existing instance and short-circuit.
@dataclass
class _Sidecar:
    name: str
    process: subprocess.Popen | None  # None ⇒ we found an external instance
    port: int


_sidecars: dict[str, _Sidecar] = {}


def _ensure_sidecar(name: str, port: int, argv: list[str]) -> _Sidecar:
    """Start ``argv`` as a daemon if nothing is already listening on ``port``."""
    if name in _sidecars:
        return _sidecars[name]
    if not is_port_available(port):
        logger.info("%s already running on port %d; reusing", name, port)
        sc = _Sidecar(name=name, process=None, port=port)
    else:
        if shutil.which(argv[0]) is None:
            raise RuntimeError(
                f"{argv[0]} is not on PATH; install it (the Dynamo "
                f"discovery layer needs {name} when --dynamo-router-mode=kv)."
            )
        logger.info("Launching %s on port %d", name, port)
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sc = _Sidecar(name=name, process=proc, port=port)

    _sidecars[name] = sc
    return sc


def _ensure_etcd_and_nats():
    """Start etcd (2379) and NATS (4222) if not already up.

    These ports are the Dynamo defaults; the frontend / workers pick them
    up implicitly when ``DYN_DISCOVERY_BACKEND=etcd`` and the NATS env var
    is unset (defaults to ``nats://localhost:4222``).
    """
    _ensure_sidecar(
        "etcd",
        2379,
        [
            "etcd",
            "--listen-client-urls", "http://0.0.0.0:2379",
            "--advertise-client-urls", "http://127.0.0.1:2379",
            "--listen-peer-urls", "http://0.0.0.0:2380",
            "--initial-advertise-peer-urls", "http://127.0.0.1:2380",
            "--initial-cluster", "default=http://127.0.0.1:2380",
            "--data-dir", "/tmp/dynamo-etcd",
        ],
    )
    _ensure_sidecar(
        "nats-server",
        4222,
        ["nats-server", "-a", "0.0.0.0", "-p", "4222"],
    )
    # Give them a beat to bind before the frontend / workers start
    # registering.  Polling /health on etcd would be cleaner, but the
    # 2-second wait keeps the code dependency-free.
    time.sleep(2)


def start_dynamo_router(
    args,
    *,
    has_pd_disaggregation: bool = False,
    force_new: bool = False,
) -> tuple[str, int]:
    """Drop-in replacement for :func:`miles.ray.rollout._start_router`.

    Same signature, same return value (``(host, port)``). The cluster-wide
    state (``args.sglang_router_ip``, ``args.sglang_router_port``) is read
    and written the same way too.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_mode = getattr(args, "dynamo_router_mode", None) or (
        "kv" if has_pd_disaggregation else "round-robin"
    )
    discovery = getattr(args, "dynamo_discovery_backend", None)
    if router_mode == "kv":
        discovery = "etcd"
    elif discovery is None:
        discovery = "file"

    if discovery == "etcd":
        _ensure_etcd_and_nats()

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    if not is_port_available(router_port):
        raise RuntimeError(
            f"Port {router_port} already bound — stale Dynamo frontend? "
            f"Run 'pkill -f dynamo.frontend' and retry."
        )

    argv: list[str] = [
        sys.executable, "-m", "dynamo.frontend",
        "--http-port", str(router_port),
        "--router-mode", router_mode,
        "--discovery-backend", discovery,
    ]

    # Older Dynamo frontends don't recognise the predict-on-route /
    # no-router-kv-events flags. Probe with --help and only pass what's
    # actually supported.
    if router_mode == "kv":
        help_text = ""
        try:
            help_text = subprocess.run(
                [sys.executable, "-m", "dynamo.frontend", "--help"],
                capture_output=True, text=True, timeout=15,
            ).stdout
        except Exception:
            pass
        if "--router-predict-on-route" in help_text and getattr(args, "dynamo_router_predict_on_route", True):
            argv += ["--router-predict-on-route"]
            ttl = getattr(args, "dynamo_router_predicted_ttl_secs", None)
            if ttl is not None and "--router-predicted-ttl-secs" in help_text:
                argv += ["--router-predicted-ttl-secs", str(ttl)]
        if "--no-router-kv-events" in help_text and not getattr(args, "dynamo_router_kv_events", True):
            argv += ["--no-router-kv-events"]

    env = os.environ.copy()
    env["DYN_DISCOVERY_BACKEND"] = discovery
    # Turn on Dynamo's RL mode on the frontend. With this set, the frontend:
    #   1. forces return_token_ids=true (drops detokenize text round-trip),
    #   2. injects prompt_token_ids into the response,
    #   3. promotes nvext.completion_token_ids onto choice.token_ids,
    #   4. forces logprobs=true.
    # Per the upstream commit message, this drops train↔rollout logprob
    # mismatch KL from ~1.0 to ~0.005. Without it our run measured ~0.03.
    env["DYN_ENABLE_RL"] = "true"

    logger.info("Starting Dynamo frontend: %s", " ".join(argv))
    proc = subprocess.Popen(argv, env=env)

    # Miles' wait_for_server_ready was written against multiprocessing.Process
    # (it calls ``.is_alive()``). subprocess.Popen exposes ``.poll()`` instead,
    # so we wrap it in a tiny shim that re-exports the right name.
    class _PopenAlive:
        def __init__(self, p): self._p = p
        def is_alive(self): return self._p.poll() is None
        @property
        def pid(self): return self._p.pid

    wait_for_server_ready(router_ip, router_port, _PopenAlive(proc), timeout=60)
    logger.info("Dynamo frontend healthy at %s:%d", router_ip, router_port)
    return router_ip, router_port
