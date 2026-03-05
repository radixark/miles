from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.utils.ft.platform.k8s_node_manager import (
    LABEL_KEY,
    K8sNodeManager,
    REASON_LABEL_KEY,
    _build_label_keys,
    query_bad_nodes,
)


def _make_manager_with_mock_api(
    label_suffix: str = "",
) -> tuple[K8sNodeManager, AsyncMock]:
    """Create a K8sNodeManager with a mocked CoreV1Api injected via ApiClient."""
    mock_api_client = MagicMock()
    manager = K8sNodeManager(api_client=mock_api_client, label_suffix=label_suffix)

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
        mock_core_v1.list_node.return_value = SimpleNamespace(
            items=[mock_node_1, mock_node_2]
        )

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

        nodes = [
            SimpleNamespace(metadata=SimpleNamespace(name=f"node-{i}"))
            for i in range(5)
        ]
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


class TestQueryBadNodesSyncHelper:
    def test_returns_node_list_on_success(self) -> None:
        with patch(
            "miles.utils.ft.platform.k8s_node_manager.K8sNodeManager"
        ) as mock_cls:
            instance = mock_cls.return_value
            async def fake_get_bad_nodes() -> list[str]:
                return ["node-a", "node-b"]
            instance.get_bad_nodes = fake_get_bad_nodes
            instance.aclose = AsyncMock()

            result = query_bad_nodes()

        assert result == ["node-a", "node-b"]

    def test_returns_none_on_exception(self) -> None:
        with patch(
            "miles.utils.ft.platform.k8s_node_manager.K8sNodeManager"
        ) as mock_cls:
            instance = mock_cls.return_value
            async def failing_get_bad_nodes() -> list[str]:
                raise ConnectionError("K8s unreachable")
            instance.get_bad_nodes = failing_get_bad_nodes
            instance.aclose = AsyncMock()

            result = query_bad_nodes()

        assert result is None

    def test_passes_label_suffix_to_manager(self) -> None:
        with patch(
            "miles.utils.ft.platform.k8s_node_manager.K8sNodeManager"
        ) as mock_cls:
            instance = mock_cls.return_value
            async def fake_get_bad_nodes() -> list[str]:
                return []
            instance.get_bad_nodes = fake_get_bad_nodes
            instance.aclose = AsyncMock()

            query_bad_nodes(label_suffix="test1")

        mock_cls.assert_called_once_with(label_suffix="test1")

    def test_reads_label_suffix_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MILES_FT_K8S_LABEL_SUFFIX", "envsfx")
        with patch(
            "miles.utils.ft.platform.k8s_node_manager.K8sNodeManager"
        ) as mock_cls:
            instance = mock_cls.return_value
            async def fake_get_bad_nodes() -> list[str]:
                return []
            instance.get_bad_nodes = fake_get_bad_nodes
            instance.aclose = AsyncMock()

            query_bad_nodes()

        mock_cls.assert_called_once_with(label_suffix="envsfx")


class TestLabelSuffix:
    def test_build_label_keys_no_suffix(self) -> None:
        label_key, reason_key = _build_label_keys("")
        assert label_key == "ft.miles.io/disabled"
        assert reason_key == "ft.miles.io/disabled-reason"

    def test_build_label_keys_with_suffix(self) -> None:
        label_key, reason_key = _build_label_keys("test1")
        assert label_key == "ft.miles.io/disabled-test1"
        assert reason_key == "ft.miles.io/disabled-reason-test1"

    @pytest.mark.anyio
    async def test_mark_node_bad_uses_suffixed_labels(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(label_suffix="sfx")

        await manager.mark_node_bad(node_id="node-1", reason="test")

        body = mock_core_v1.patch_node.call_args.kwargs["body"]
        assert "ft.miles.io/disabled-sfx" in body["metadata"]["labels"]
        assert "ft.miles.io/disabled-reason-sfx" in body["metadata"]["labels"]

    @pytest.mark.anyio
    async def test_get_bad_nodes_uses_suffixed_selector(self) -> None:
        manager, mock_core_v1 = _make_manager_with_mock_api(label_suffix="sfx")
        mock_core_v1.list_node.return_value = SimpleNamespace(items=[])

        await manager.get_bad_nodes()

        mock_core_v1.list_node.assert_awaited_once_with(
            label_selector="ft.miles.io/disabled-sfx=true",
        )
