# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator

import pytest
from kubernetes_asyncio import client as kubernetes_client
from tests.e2e.k8s_apiserver.apiserver import (
    ApiserverEnvironment,
    log_apiserver_diagnostics,
    start_apiserver,
    stop_apiserver,
)
from tests.e2e.k8s_apiserver.environment import keep_environment, new_run_id, require_docker
from tests.e2e.k8s_apiserver.utils import build_insecure_api_client, create_namespace, wait_until_serving

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def k8s_run_id() -> str:
    require_docker()
    return new_run_id()


@pytest.fixture(scope="session")
def apiserver_environment(k8s_run_id: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[ApiserverEnvironment]:
    environment = start_apiserver(run_id=f"{k8s_run_id}-api", work_dir=tmp_path_factory.mktemp("apiserver"))
    try:
        asyncio.run(_wait_until_apiserver_serves(environment))
        yield environment
    finally:
        if keep_environment():
            logger.warning(f"leaving the apiserver running as requested {environment=}")
        else:
            stop_apiserver(environment)


@pytest.fixture
async def apiserver_core_v1(
    apiserver_environment: ApiserverEnvironment,
) -> AsyncIterator[kubernetes_client.CoreV1Api]:
    api_client = build_insecure_api_client(endpoint=apiserver_environment.endpoint, token=apiserver_environment.token)
    try:
        yield kubernetes_client.CoreV1Api(api_client)
    finally:
        await api_client.close()


@pytest.fixture
async def apiserver_namespace(apiserver_core_v1: kubernetes_client.CoreV1Api) -> str:
    return await create_namespace(apiserver_core_v1)


@pytest.fixture(scope="session")
def expiring_apiserver_environment(
    k8s_run_id: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[ApiserverEnvironment]:
    environment = start_apiserver(
        run_id=f"{k8s_run_id}-expiry", work_dir=tmp_path_factory.mktemp("apiserver-expiry"), watch_cache=False
    )
    try:
        asyncio.run(_wait_until_apiserver_serves(environment))
        yield environment
    finally:
        if keep_environment():
            logger.warning(f"leaving the apiserver running as requested {environment=}")
        else:
            stop_apiserver(environment)


@pytest.fixture
async def expiring_apiserver_core_v1(
    expiring_apiserver_environment: ApiserverEnvironment,
) -> AsyncIterator[kubernetes_client.CoreV1Api]:
    api_client = build_insecure_api_client(
        endpoint=expiring_apiserver_environment.endpoint, token=expiring_apiserver_environment.token
    )
    try:
        yield kubernetes_client.CoreV1Api(api_client)
    finally:
        await api_client.close()


@pytest.fixture
async def expiring_apiserver_namespace(expiring_apiserver_core_v1: kubernetes_client.CoreV1Api) -> str:
    return await create_namespace(expiring_apiserver_core_v1)


async def _wait_until_apiserver_serves(environment: ApiserverEnvironment) -> None:
    api_client = build_insecure_api_client(endpoint=environment.endpoint, token=environment.token)
    try:
        await wait_until_serving(kubernetes_client.CoreV1Api(api_client))
    except BaseException:
        log_apiserver_diagnostics(environment)
        raise
    finally:
        await api_client.close()
