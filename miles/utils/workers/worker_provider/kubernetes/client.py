from __future__ import annotations

import asyncio

from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi


def create_kubernetes_client() -> KubernetesAsyncioPodApi:
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    return KubernetesAsyncioPodApi(core_v1_api=kubernetes_client.CoreV1Api(kubernetes_client.ApiClient()))


async def delete_pods(*, namespace: str, pod_names: list[str]) -> None:
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    async with kubernetes_client.ApiClient() as api_client:
        core_v1_api = kubernetes_client.CoreV1Api(api_client)
        await asyncio.gather(
            *(core_v1_api.delete_namespaced_pod(name=pod_name, namespace=namespace) for pod_name in pod_names)
        )
