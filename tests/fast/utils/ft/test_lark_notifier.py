import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier


class TestLarkWebhookNotifier:
    @pytest.fixture
    async def notifier(self) -> LarkWebhookNotifier:
        instance = LarkWebhookNotifier(webhook_url="https://open.larksuite.com/open-apis/bot/v2/hook/test-token")
        yield instance
        await instance.aclose()

    @pytest.mark.asyncio
    async def test_send_posts_correct_json(self, notifier: LarkWebhookNotifier) -> None:
        mock_response = httpx.Response(status_code=200, request=httpx.Request("POST", "https://example.com"))
        with patch.object(notifier._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await notifier.send(title="Fault Alert", content="GPU lost on node-3", severity="critical")

            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            assert payload["msg_type"] == "interactive"
            assert payload["card"]["header"]["title"]["tag"] == "plain_text"
            assert payload["card"]["header"]["title"]["content"] == "[critical] Fault Alert"
            assert payload["card"]["elements"] == [{"tag": "markdown", "content": "GPU lost on node-3"}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("severity", ["critical", "warning", "info"])
    async def test_send_includes_severity_in_header(
        self, notifier: LarkWebhookNotifier, severity: str,
    ) -> None:
        mock_response = httpx.Response(status_code=200, request=httpx.Request("POST", "https://example.com"))
        with patch.object(notifier._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await notifier.send(title="Alert", content="test", severity=severity)

            payload = mock_post.call_args[1]["json"]
            assert payload["card"]["header"]["title"]["content"].startswith(f"[{severity}]")

    @pytest.mark.asyncio
    async def test_send_http_error_does_not_raise(
        self, notifier: LarkWebhookNotifier, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_response = httpx.Response(status_code=500, request=httpx.Request("POST", "https://example.com"))
        with patch.object(notifier._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with caplog.at_level(logging.WARNING):
                await notifier.send(title="Fault Alert", content="test error", severity="critical")

            assert "lark_webhook_send_failed" in caplog.text

    @pytest.mark.asyncio
    async def test_send_connect_error_does_not_raise(
        self, notifier: LarkWebhookNotifier, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch.object(
            notifier._client, "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            with caplog.at_level(logging.WARNING):
                await notifier.send(title="Alert", content="unreachable", severity="warning")

            assert "lark_webhook_send_failed" in caplog.text
