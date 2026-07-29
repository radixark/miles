# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable

from kubernetes_asyncio import client as kubernetes_client

logger = logging.getLogger(__name__)

CELL_LABEL = "miles-cell"
PAUSE_IMAGE = "registry.k8s.io/pause:3.9"
BUSYBOX_IMAGE = "busybox:1.36"


def unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def pod_body(
    *,
    name: str,
    cell: str,
    image: str = PAUSE_IMAGE,
    command: list[str] | None = None,
    restart_policy: str = "Never",
    grace_period_seconds: int = 0,
) -> kubernetes_client.V1Pod:
    return kubernetes_client.V1Pod(
        metadata=kubernetes_client.V1ObjectMeta(name=name, labels={CELL_LABEL: cell}),
        spec=kubernetes_client.V1PodSpec(
            restart_policy=restart_policy,
            termination_grace_period_seconds=grace_period_seconds,
            containers=[kubernetes_client.V1Container(name="main", image=image, command=command)],
        ),
    )


def build_insecure_api_client(*, endpoint: str, token: str) -> kubernetes_client.ApiClient:
    configuration = kubernetes_client.Configuration(host=endpoint)
    configuration.verify_ssl = False
    api_client = kubernetes_client.ApiClient(configuration=configuration)
    api_client.set_default_header("Authorization", f"Bearer {token}")
    return api_client


async def create_namespace(core_v1_api: kubernetes_client.CoreV1Api) -> str:
    namespace = unique_name("miles-test")
    await core_v1_api.create_namespace(
        body=kubernetes_client.V1Namespace(metadata=kubernetes_client.V1ObjectMeta(name=namespace))
    )
    return namespace


async def wait_until_serving(core_v1_api: kubernetes_client.CoreV1Api, *, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            await core_v1_api.list_namespace(limit=1)
            return
        except Exception:
            if time.monotonic() >= deadline:
                logger.error(f"the apiserver never started serving within {timeout=}", exc_info=True)
                raise
            await asyncio.sleep(0.5)


async def wait_until(
    predicate: Callable[[], bool], *, description: str, timeout: float = 120.0, interval: float = 0.2
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out after {timeout}s waiting until {description}"
        await asyncio.sleep(interval)
