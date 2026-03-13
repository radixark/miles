from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.utils.ft.adapters.impl.k8s_node_manager import (
    LABEL_KEY,
    REASON_LABEL_KEY,
    K8sNodeManager,
    _build_label_keys,
)


def _make_manager_with_mock_api(
    label_prefix: str = "",
    ray_cluster_name: str = "",
    namespace: str = "default",
) -> tuple[K8sNodeManager, AsyncMock]:
    """Create a K8sNodeManager with a mocked CoreV1Api injected via ApiClient."""
    mock_api_client = MagicMock()
    manager = K8sNodeManager(
        api_client=mock_api_client,
        label_prefix=label_prefix,
        namespace=namespace,
    )

    if ray_cluster_name:
        manager._ray_cluster_name = ray_cluster_name

    mock_core_v1 = AsyncMock()
    manager._ensure_client = AsyncMock(return_value=mock_core_v1)
    manager._ensure_client_unlocked = AsyncMock(return_value=mock_core_v1)
    manager._affinity_validated = True
    return manager, mock_core_v1


class TestMarkNodeBad:
    @pytest.mark.anyio
    async def test_patches_node_with_correct_labels(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")

        await manager.mark_node_bad(node_id="node-1", reason="gpu_ecc_error")

        mock_core_v1.patch_node.assert_awaited_once()
        call_kwargs = mock_core_v1.patch_node.call_args
        assert call_kwargs.kwargs["name"] == "node-1"
        body = call_kwargs.kwargs["body"]
        assert body["metadata"]["labels"][LABEL_KEY] == "true"
        assert body["metadata"]["labels"][REASON_LABEL_KEY] == "gpu_ecc_error"

    @pytest.mark.anyio
    async def test_raises_on_api_failure(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        mock_core_v1.patch_node.side_effect = Exception("K8s API unreachable")

        with pytest.raises(Exception, match="K8s API unreachable"):
            await manager.mark_node_bad(node_id="node-1", reason="test")


class TestLabelPrefix:
    def test_build_label_keys_no_prefix(self) -> None:
        label_key, reason_key = _build_label_keys("")
        assert label_key == "ft.miles.io/disabled"
        assert reason_key == "ft.miles.io/disabled-reason"

    def test_build_label_keys_with_prefix(self) -> None:
        label_key, reason_key = _build_label_keys("test1")
        assert label_key == "ft.miles.io/test1-disabled"
        assert reason_key == "ft.miles.io/test1-disabled-reason"

    @pytest.mark.anyio
    async def test_mark_node_bad_uses_prefixed_labels(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(
            label_prefix="pfx",
            ray_cluster_name="c1",
        )

        await manager.mark_node_bad(node_id="node-1", reason="test")

        body = mock_core_v1.patch_node.call_args.kwargs["body"]
        assert "ft.miles.io/pfx-disabled" in body["metadata"]["labels"]
        assert "ft.miles.io/pfx-disabled-reason" in body["metadata"]["labels"]


class TestMarkNodeBadDeletesPod:
    @pytest.mark.anyio
    async def test_mark_node_bad_deletes_worker_pod(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(
            ray_cluster_name="my-cluster",
            namespace="test-ns",
        )
        mock_pod = SimpleNamespace(metadata=SimpleNamespace(name="my-cluster-worker-abc"))
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[mock_pod])

        await manager.mark_node_bad(node_id="node-1", reason="gpu_ecc_error")

        mock_core_v1.patch_node.assert_awaited_once()
        mock_core_v1.list_namespaced_pod.assert_awaited_once()
        list_kwargs = mock_core_v1.list_namespaced_pod.call_args.kwargs
        assert list_kwargs["namespace"] == "test-ns"
        assert "ray.io/cluster=my-cluster" in list_kwargs["label_selector"]
        assert "ray.io/node-type=worker" in list_kwargs["label_selector"]
        assert list_kwargs["field_selector"] == "spec.nodeName=node-1"

        mock_core_v1.delete_namespaced_pod.assert_awaited_once()
        del_kwargs = mock_core_v1.delete_namespaced_pod.call_args.kwargs
        assert del_kwargs["name"] == "my-cluster-worker-abc"
        assert del_kwargs["namespace"] == "test-ns"

    @pytest.mark.anyio
    async def test_mark_node_bad_handles_no_pods_on_node(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="cluster-x")
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[])

        await manager.mark_node_bad(node_id="node-1", reason="test")

        mock_core_v1.patch_node.assert_awaited_once()
        mock_core_v1.list_namespaced_pod.assert_awaited_once()
        mock_core_v1.delete_namespaced_pod.assert_not_awaited()

    @pytest.mark.anyio
    async def test_pod_delete_failure_logs_but_does_not_raise(self) -> None:
        """Pod delete failure is caught and logged; mark_node_bad still succeeds."""
        manager, mock_core_v1 = _make_manager_with_mock_api(
            ray_cluster_name="my-cluster",
            namespace="test-ns",
        )
        manager._delete_ray_worker_pod_on_node = AsyncMock(
            side_effect=Exception("pod delete timeout"),
        )

        await manager.mark_node_bad(node_id="node-1", reason="gpu_ecc_error")

        mock_core_v1.patch_node.assert_awaited_once()
        manager._delete_ray_worker_pod_on_node.assert_awaited_once_with("node-1")

    @pytest.mark.anyio
    async def test_mark_node_bad_deletes_multiple_pods(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        pods = [
            SimpleNamespace(metadata=SimpleNamespace(name="c1-worker-0")),
            SimpleNamespace(metadata=SimpleNamespace(name="c1-worker-1")),
        ]
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=pods)

        await manager.mark_node_bad(node_id="node-1", reason="test")

        assert mock_core_v1.delete_namespaced_pod.await_count == 2
        deleted_names = [call.kwargs["name"] for call in mock_core_v1.delete_namespaced_pod.call_args_list]
        assert "c1-worker-0" in deleted_names
        assert "c1-worker-1" in deleted_names


class TestDeleteRayWorkerPodOnNode:
    @pytest.mark.anyio
    async def test_uses_correct_selectors(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(
            ray_cluster_name="train-cluster",
            namespace="ml",
        )
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[])

        await manager._delete_ray_worker_pod_on_node("gpu-node-3")

        mock_core_v1.list_namespaced_pod.assert_awaited_once()
        kwargs = mock_core_v1.list_namespaced_pod.call_args.kwargs
        assert kwargs["namespace"] == "ml"
        assert kwargs["label_selector"] == "ray.io/cluster=train-cluster,ray.io/node-type=worker"
        assert kwargs["field_selector"] == "spec.nodeName=gpu-node-3"


class TestMetadataTranslation:
    @pytest.mark.anyio
    async def test_mark_node_bad_uses_k8s_name_from_metadata(self) -> None:
        """mark_node_bad uses k8s_node_name from metadata to call K8s API."""
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")

        await manager.mark_node_bad(
            node_id="ray-uuid-abc",
            reason="gpu_ecc",
            node_metadata={"k8s_node_name": "gke-node-01"},
        )

        call_kwargs = mock_core_v1.patch_node.call_args
        assert call_kwargs.kwargs["name"] == "gke-node-01"

    @pytest.mark.anyio
    async def test_mark_node_bad_without_metadata_uses_node_id(self) -> None:
        """Without metadata, mark_node_bad falls back to node_id."""
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")

        await manager.mark_node_bad(node_id="ray-uuid-abc", reason="test")

        call_kwargs = mock_core_v1.patch_node.call_args
        assert call_kwargs.kwargs["name"] == "ray-uuid-abc"

    @pytest.mark.anyio
    async def test_mark_node_bad_with_metadata_deletes_pod_using_k8s_name(self) -> None:
        """When metadata provides k8s_node_name, pod deletion uses that name."""
        manager, mock_core_v1 = _make_manager_with_mock_api(
            ray_cluster_name="my-cluster",
            namespace="test-ns",
        )
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[])

        await manager.mark_node_bad(
            node_id="ray-uuid-abc",
            reason="test",
            node_metadata={"k8s_node_name": "gke-node-01"},
        )

        list_kwargs = mock_core_v1.list_namespaced_pod.call_args.kwargs
        assert list_kwargs["field_selector"] == "spec.nodeName=gke-node-01"


def _make_mock_pod_with_affinity(
    label_key: str = "ft.miles.io/disabled",
    operator: str = "NotIn",
    values: list[str] | None = None,
) -> SimpleNamespace:
    """Build a mock pod with nodeAffinity match_expressions."""
    if values is None:
        values = ["true"]
    return SimpleNamespace(
        metadata=SimpleNamespace(name="worker-0"),
        spec=SimpleNamespace(
            affinity=SimpleNamespace(
                node_affinity=SimpleNamespace(
                    required_during_scheduling_ignored_during_execution=SimpleNamespace(
                        node_selector_terms=[
                            SimpleNamespace(
                                match_expressions=[
                                    SimpleNamespace(key=label_key, operator=operator, values=values),
                                ]
                            ),
                        ],
                    ),
                ),
            ),
        ),
    )


class TestAssertWorkerNodeAffinity:
    @pytest.mark.anyio
    async def test_passes_when_affinity_correct(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        mock_pod = _make_mock_pod_with_affinity()
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[mock_pod])

        await manager.assert_worker_node_affinity()

    @pytest.mark.anyio
    async def test_raises_when_affinity_missing(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        mock_pod = SimpleNamespace(
            metadata=SimpleNamespace(name="worker-0"),
            spec=SimpleNamespace(affinity=None),
        )
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[mock_pod])

        with pytest.raises(RuntimeError, match="missing nodeAffinity NotIn rule"):
            await manager.assert_worker_node_affinity()

    @pytest.mark.anyio
    async def test_raises_when_wrong_label_key(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        mock_pod = _make_mock_pod_with_affinity(label_key="wrong.io/label")
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[mock_pod])

        with pytest.raises(RuntimeError, match="missing nodeAffinity NotIn rule"):
            await manager.assert_worker_node_affinity()

    @pytest.mark.anyio
    async def test_raises_when_no_worker_pods(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[])

        with pytest.raises(RuntimeError, match="no ray worker pods found"):
            await manager.assert_worker_node_affinity()

    @pytest.mark.anyio
    async def test_first_pod_ok_second_pod_missing_affinity_raises(self) -> None:
        """The old implementation only checked pod_list.items[0], so when
        the first pod was correct but the second was missing the affinity
        rule, it would incorrectly pass. Now all pods are checked."""
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        good_pod = _make_mock_pod_with_affinity()
        bad_pod = SimpleNamespace(
            metadata=SimpleNamespace(name="worker-bad"),
            spec=SimpleNamespace(affinity=None),
        )
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[good_pod, bad_pod])

        with pytest.raises(RuntimeError, match="worker-bad"):
            await manager.assert_worker_node_affinity()

    @pytest.mark.anyio
    async def test_all_pods_correct_passes(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        pods = [_make_mock_pod_with_affinity() for _ in range(3)]
        for i, pod in enumerate(pods):
            pod.metadata.name = f"worker-{i}"
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=pods)

        await manager.assert_worker_node_affinity()

    @pytest.mark.anyio
    async def test_mark_node_bad_triggers_assertion_once(self) -> None:
        """First mark_node_bad calls assert_worker_node_affinity; second does not."""
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        manager._affinity_validated = False

        mock_pod = _make_mock_pod_with_affinity()
        mock_core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[mock_pod])

        await manager.mark_node_bad(node_id="node-1", reason="test")

        # Step 1: list_namespaced_pod called twice — once for affinity check, once for pod delete
        assert mock_core_v1.list_namespaced_pod.await_count == 2

        mock_core_v1.list_namespaced_pod.reset_mock()
        await manager.mark_node_bad(node_id="node-2", reason="test")

        # Step 2: only pod delete call, no second affinity check
        assert mock_core_v1.list_namespaced_pod.await_count == 1


class TestAutoDetectRayClusterName:
    @pytest.mark.anyio
    async def test_detects_cluster_name_from_pod_labels(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(namespace="test-ns")
        mock_pod = SimpleNamespace(
            metadata=SimpleNamespace(
                labels={"ray.io/cluster": "my-train-cluster", "app": "ray"},
            ),
        )
        mock_core_v1.read_namespaced_pod.return_value = mock_pod

        with patch.dict("os.environ", {"K8S_POD_NAME": "my-pod-0"}):
            result = await manager._ensure_ray_cluster_name()

        assert result == "my-train-cluster"
        assert manager._ray_cluster_name == "my-train-cluster"
        mock_core_v1.read_namespaced_pod.assert_awaited_once()
        call_kwargs = mock_core_v1.read_namespaced_pod.call_args.kwargs
        assert call_kwargs["name"] == "my-pod-0"
        assert call_kwargs["namespace"] == "test-ns"

    @pytest.mark.anyio
    async def test_raises_when_pod_name_env_missing(self) -> None:
        manager, _ = _make_manager_with_mock_api()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="K8S_POD_NAME env var not set"):
                await manager._ensure_ray_cluster_name()

    @pytest.mark.anyio
    async def test_raises_when_pod_missing_cluster_label(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()
        mock_pod = SimpleNamespace(
            metadata=SimpleNamespace(labels={"app": "ray"}),
        )
        mock_core_v1.read_namespaced_pod.return_value = mock_pod

        with patch.dict("os.environ", {"K8S_POD_NAME": "my-pod-0"}):
            with pytest.raises(RuntimeError, match="missing ray.io/cluster label"):
                await manager._ensure_ray_cluster_name()

    @pytest.mark.anyio
    async def test_caches_detected_cluster_name(self) -> None:
        """Second call returns cached value without querying the API again."""
        manager, mock_core_v1 = _make_manager_with_mock_api(namespace="ns")
        mock_pod = SimpleNamespace(
            metadata=SimpleNamespace(labels={"ray.io/cluster": "cached-cluster"}),
        )
        mock_core_v1.read_namespaced_pod.return_value = mock_pod

        with patch.dict("os.environ", {"K8S_POD_NAME": "pod-1"}):
            first = await manager._ensure_ray_cluster_name()
            second = await manager._ensure_ray_cluster_name()

        assert first == second == "cached-cluster"
        assert mock_core_v1.read_namespaced_pod.await_count == 1


class TestInitLockPreventsDuplicateWork:
    """_ensure_ray_cluster_name and the affinity check had check-then-act races:
    two concurrent mark_node_bad calls could both see empty state, both execute
    the K8s API query, and the first ApiClient would leak.
    Fix: asyncio.Lock in _ensure_ray_cluster_name and mark_node_bad."""

    @pytest.mark.anyio
    async def test_concurrent_ensure_ray_cluster_name_calls_api_once(self) -> None:
        """Two concurrent _ensure_ray_cluster_name calls should only query the
        K8s API once thanks to the asyncio.Lock."""
        manager, mock_core_v1 = _make_manager_with_mock_api(namespace="ns")
        mock_pod = SimpleNamespace(
            metadata=SimpleNamespace(labels={"ray.io/cluster": "deduped-cluster"}),
        )

        async def _slow_read_pod(**kwargs: object) -> SimpleNamespace:
            await asyncio.sleep(0.01)
            return mock_pod

        mock_core_v1.read_namespaced_pod.side_effect = _slow_read_pod

        with patch.dict("os.environ", {"K8S_POD_NAME": "pod-x"}):
            results = await asyncio.gather(
                manager._ensure_ray_cluster_name(),
                manager._ensure_ray_cluster_name(),
            )

        assert results == ["deduped-cluster", "deduped-cluster"]
        assert mock_core_v1.read_namespaced_pod.await_count == 1

    @pytest.mark.anyio
    async def test_concurrent_mark_node_bad_validates_affinity_once(self) -> None:
        """Two concurrent mark_node_bad calls should only run affinity validation once."""
        manager, mock_core_v1 = _make_manager_with_mock_api(ray_cluster_name="c1")
        manager._affinity_validated = False

        mock_pod = _make_mock_pod_with_affinity()

        affinity_call_count = 0

        async def _counting_list(**kwargs: object) -> SimpleNamespace:
            nonlocal affinity_call_count
            if "affinity_check" not in str(kwargs):
                pass
            affinity_call_count += 1
            await asyncio.sleep(0.01)
            return SimpleNamespace(items=[mock_pod])

        mock_core_v1.list_namespaced_pod.side_effect = _counting_list

        await asyncio.gather(
            manager.mark_node_bad(node_id="node-1", reason="test1"),
            manager.mark_node_bad(node_id="node-2", reason="test2"),
        )

        assert manager._affinity_validated is True

    @pytest.mark.anyio
    async def test_concurrent_ensure_client_creates_api_client_once(self) -> None:
        """Previously _ensure_client was not protected by the init lock, so
        two concurrent callers could both see _core_v1 is None and both create
        an ApiClient, leaking the first one. Now _ensure_client acquires the
        lock, and the real init logic is in _ensure_client_unlocked."""
        manager = K8sNodeManager(namespace="default")
        manager._api_client = MagicMock()

        results = await asyncio.gather(
            manager._ensure_client(),
            manager._ensure_client(),
        )

        assert results[0] is results[1]
