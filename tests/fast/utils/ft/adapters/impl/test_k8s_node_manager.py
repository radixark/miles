from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.utils.ft.adapters.impl.k8s_node_manager import LABEL_KEY, REASON_LABEL_KEY, K8sNodeManager, _build_label_keys


def _make_manager_with_mock_api(
    label_prefix: str = "",
) -> tuple[K8sNodeManager, AsyncMock]:
    """Create a K8sNodeManager with a mocked CoreV1Api injected via ApiClient."""
    mock_api_client = MagicMock()
    manager = K8sNodeManager(api_client=mock_api_client, label_prefix=label_prefix)

    mock_core_v1 = AsyncMock()
    manager._ensure_client = AsyncMock(return_value=mock_core_v1)
    return manager, mock_core_v1


class TestMarkNodeBad:
    @pytest.mark.anyio
    async def test_patches_node_with_correct_labels(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()

        await manager.mark_node_bad(node_id="node-1", reason="gpu_ecc_error")

        mock_core_v1.patch_node.assert_awaited_once()
        call_kwargs = mock_core_v1.patch_node.call_args
        assert call_kwargs.kwargs["name"] == "node-1"
        body = call_kwargs.kwargs["body"]
        assert body["metadata"]["labels"][LABEL_KEY] == "true"
        assert body["metadata"]["labels"][REASON_LABEL_KEY] == "gpu_ecc_error"

    @pytest.mark.anyio
    async def test_raises_on_api_failure(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()
        mock_core_v1.patch_node.side_effect = Exception("K8s API unreachable")

        with pytest.raises(Exception, match="K8s API unreachable"):
            await manager.mark_node_bad(node_id="node-1", reason="test")


class TestUnmarkNodeBad:
    @pytest.mark.anyio
    async def test_patches_node_with_none_labels(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()

        await manager.unmark_node_bad(node_id="node-2")

        mock_core_v1.patch_node.assert_awaited_once()
        call_kwargs = mock_core_v1.patch_node.call_args
        assert call_kwargs.kwargs["name"] == "node-2"
        body = call_kwargs.kwargs["body"]
        assert body["metadata"]["labels"][LABEL_KEY] is None
        assert body["metadata"]["labels"][REASON_LABEL_KEY] is None

    @pytest.mark.anyio
    async def test_raises_on_api_failure(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()
        mock_core_v1.patch_node.side_effect = Exception("K8s API unreachable")

        with pytest.raises(Exception, match="K8s API unreachable"):
            await manager.unmark_node_bad(node_id="node-2")


class TestGetBadNodes:
    @pytest.mark.anyio
    async def test_returns_node_names_with_label(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()

        mock_node_1 = SimpleNamespace(metadata=SimpleNamespace(name="node-a"))
        mock_node_2 = SimpleNamespace(metadata=SimpleNamespace(name="node-b"))
        mock_core_v1.list_node.return_value = SimpleNamespace(items=[mock_node_1, mock_node_2])

        result = await manager.get_bad_nodes()

        assert result == ["node-a", "node-b"]
        mock_core_v1.list_node.assert_awaited_once_with(
            label_selector=f"{LABEL_KEY}=true",
        )

    @pytest.mark.anyio
    async def test_returns_empty_list_when_no_bad_nodes(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()
        mock_core_v1.list_node.return_value = SimpleNamespace(items=[])

        result = await manager.get_bad_nodes()

        assert result == []

    @pytest.mark.anyio
    async def test_returns_multiple_bad_nodes(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()

        nodes = [SimpleNamespace(metadata=SimpleNamespace(name=f"node-{i}")) for i in range(5)]
        mock_core_v1.list_node.return_value = SimpleNamespace(items=nodes)

        result = await manager.get_bad_nodes()

        assert len(result) == 5
        assert result == [f"node-{i}" for i in range(5)]

    @pytest.mark.anyio
    async def test_raises_on_api_failure(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api()
        mock_core_v1.list_node.side_effect = Exception("K8s API unreachable")

        with pytest.raises(Exception, match="K8s API unreachable"):
            await manager.get_bad_nodes()


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
        manager, mock_core_v1 = _make_manager_with_mock_api(label_prefix="pfx")

        await manager.mark_node_bad(node_id="node-1", reason="test")

        body = mock_core_v1.patch_node.call_args.kwargs["body"]
        assert "ft.miles.io/pfx-disabled" in body["metadata"]["labels"]
        assert "ft.miles.io/pfx-disabled-reason" in body["metadata"]["labels"]

    @pytest.mark.anyio
    async def test_get_bad_nodes_uses_prefixed_selector(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(label_prefix="pfx")
        mock_core_v1.list_node.return_value = SimpleNamespace(items=[])

        await manager.get_bad_nodes()

        mock_core_v1.list_node.assert_awaited_once_with(
            label_selector="ft.miles.io/pfx-disabled=true",
        )
