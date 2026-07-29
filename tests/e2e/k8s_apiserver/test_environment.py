# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
from typing import Any

from kubernetes_asyncio import client as kubernetes_client
from kubernetes_asyncio import watch as kubernetes_watch
from tests.ci.ci_register import register_cpu_ci
from tests.e2e.k8s_apiserver.apiserver import (
    ApiserverEnvironment,
    apiserver_started_at,
    compact_etcd_to_head,
    restart_apiserver,
)
from tests.e2e.k8s_apiserver.utils import CELL_LABEL, create_namespace, pod_body, wait_until, wait_until_serving

register_cpu_ci(est_time=180, suite="stage-b-cpu", labels=[])


class TestApiserverEnvironment:
    async def test_pods_round_trip_through_the_api(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        """A pod can be created, listed, and deleted without any kubelet in the environment."""
        await apiserver_core_v1.create_namespaced_pod(
            namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )
        listed = await apiserver_core_v1.list_namespaced_pod(namespace=apiserver_namespace, label_selector=CELL_LABEL)
        assert [pod.metadata.name for pod in listed.items] == ["pod-a"]

        await apiserver_core_v1.delete_namespaced_pod(namespace=apiserver_namespace, name="pod-a")
        relisted = await apiserver_core_v1.list_namespaced_pod(
            namespace=apiserver_namespace, label_selector=CELL_LABEL
        )
        assert relisted.items == []

    async def test_a_watch_delivers_events_written_after_it_opened(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        """The LIST-then-WATCH handoff works: an event after the list's resourceVersion arrives."""
        listed = await apiserver_core_v1.list_namespaced_pod(namespace=apiserver_namespace, label_selector=CELL_LABEL)

        events: list[dict[str, Any]] = []
        collector = asyncio.create_task(
            _collect_watch_events(
                apiserver_core_v1,
                namespace=apiserver_namespace,
                resource_version=listed.metadata.resource_version,
                events=events,
            )
        )
        try:
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
            )
            await wait_until(lambda: len(events) >= 1, description="the creation to reach the watch")
        finally:
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)

        assert events[0]["type"] == "ADDED"
        assert events[0]["object"].metadata.name == "pod-a"
        assert int(events[0]["object"].metadata.resource_version) > int(listed.metadata.resource_version)

    async def test_the_apiserver_survives_a_restart(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_environment: ApiserverEnvironment
    ) -> None:
        """A docker restart of the apiserver keeps the endpoint and the data in etcd."""
        survivor = await create_namespace(apiserver_core_v1)
        started_at = apiserver_started_at(apiserver_environment)

        restart_apiserver(apiserver_environment)
        await wait_until_serving(apiserver_core_v1)

        assert apiserver_started_at(apiserver_environment) != started_at
        listed = await apiserver_core_v1.list_namespace()
        assert survivor in [item.metadata.name for item in listed.items]


class TestExpiringApiserverEnvironment:
    async def test_compaction_makes_an_old_cursor_unusable(
        self,
        expiring_apiserver_core_v1: kubernetes_client.CoreV1Api,
        expiring_apiserver_namespace: str,
        expiring_apiserver_environment: ApiserverEnvironment,
    ) -> None:
        """With the watch cache off, compacting etcd to head invalidates a live cursor for real."""
        listed = await expiring_apiserver_core_v1.list_namespaced_pod(
            namespace=expiring_apiserver_namespace, label_selector=CELL_LABEL
        )
        stale_cursor = listed.metadata.resource_version

        for index in range(3):
            body = pod_body(name=f"pod-filler-{index}", cell="cell-ignored")
            body.metadata.labels = {}
            await expiring_apiserver_core_v1.create_namespaced_pod(namespace=expiring_apiserver_namespace, body=body)
        compact_etcd_to_head(expiring_apiserver_environment)

        outcome = await _first_watch_outcome(
            expiring_apiserver_core_v1,
            namespace=expiring_apiserver_namespace,
            resource_version=stale_cursor,
        )
        assert _looks_like_a_410(outcome), f"a compacted cursor was not rejected {outcome=}"


async def _collect_watch_events(
    core_v1_api: kubernetes_client.CoreV1Api, *, namespace: str, resource_version: str, events: list[dict[str, Any]]
) -> None:
    watcher = kubernetes_watch.Watch()
    async for event in watcher.stream(
        core_v1_api.list_namespaced_pod,
        namespace=namespace,
        label_selector=CELL_LABEL,
        resource_version=resource_version,
        timeout_seconds=120,
    ):
        events.append(event)


async def _first_watch_outcome(
    core_v1_api: kubernetes_client.CoreV1Api, *, namespace: str, resource_version: str
) -> Any:
    watcher = kubernetes_watch.Watch()
    try:
        async for event in watcher.stream(
            core_v1_api.list_namespaced_pod,
            namespace=namespace,
            label_selector=CELL_LABEL,
            resource_version=resource_version,
            timeout_seconds=30,
        ):
            return event
        return None
    except Exception as exception:
        return exception
    finally:
        await watcher.close()


def _looks_like_a_410(outcome: Any) -> bool:
    if isinstance(outcome, BaseException):
        status = getattr(outcome, "status", None)
        return status in (410, "410") or getattr(outcome, "code", None) == 410
    if isinstance(outcome, dict) and outcome["type"] == "ERROR":
        obj = outcome["object"]
        if isinstance(obj, dict):
            return obj.get("code") == 410 or obj.get("reason") == "Expired"
        return getattr(obj, "code", None) == 410 or getattr(obj, "reason", None) == "Expired"
    return False
