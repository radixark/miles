# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator

import pytest
from kubernetes_asyncio import client as kubernetes_client
from tests.e2e.k8s_apiserver.environment import existing_kubeconfig, keep_environment, new_run_id, require_docker
from tests.e2e.k8s_apiserver.utils import create_namespace
from tests.e2e.k8s_kind.kind_cluster import KindCluster, create_cluster, delete_cluster
from tests.e2e.k8s_kind.utils import build_kubeconfig_api_client

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def kind_cluster(tmp_path_factory: pytest.TempPathFactory) -> Iterator[KindCluster]:
    reused = existing_kubeconfig()
    if reused is not None:
        logger.warning(f"reusing the cluster the environment points at {reused=}")
        yield KindCluster(name="reused", kubeconfig=reused)
        return

    require_docker()
    cluster = create_cluster(run_id=new_run_id(), kubeconfig=tmp_path_factory.mktemp("kind") / "kubeconfig")
    try:
        yield cluster
    finally:
        if keep_environment():
            logger.warning(f"leaving the cluster up as requested {cluster=}")
        else:
            delete_cluster(cluster)


@pytest.fixture
async def cluster_core_v1(kind_cluster: KindCluster) -> AsyncIterator[kubernetes_client.CoreV1Api]:
    api_client = await build_kubeconfig_api_client(kubeconfig=kind_cluster.kubeconfig)
    try:
        yield kubernetes_client.CoreV1Api(api_client)
    finally:
        await api_client.close()


@pytest.fixture
async def cluster_namespace(cluster_core_v1: kubernetes_client.CoreV1Api) -> AsyncIterator[str]:
    namespace = await create_namespace(cluster_core_v1)
    try:
        yield namespace
    finally:
        await cluster_core_v1.delete_namespace(name=namespace, grace_period_seconds=0)
