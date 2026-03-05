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
    async def test_send_http_error_raises_after_retries(
        self, notifier: LarkWebhookNotifier, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_response = httpx.Response(status_code=500, request=httpx.Request("POST", "https://example.com"))
        with patch.object(notifier._client, "post", new_callable=AsyncMock, return_value=mock_response), \
             patch("miles.utils.ft.platform.lark_notifier.asyncio.sleep", new_callable=AsyncMock):
            with caplog.at_level(logging.WARNING), pytest.raises(httpx.HTTPStatusError):
                await notifier.send(title="Fault Alert", content="test error", severity="critical")

            assert "lark_webhook_send_failed" in caplog.text

    @pytest.mark.asyncio
    async def test_send_connect_error_raises_after_retries(
        self, notifier: LarkWebhookNotifier, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch.object(
            notifier._client, "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ), patch("miles.utils.ft.platform.lark_notifier.asyncio.sleep", new_callable=AsyncMock):
            with caplog.at_level(logging.WARNING), pytest.raises(httpx.ConnectError):
                await notifier.send(title="Alert", content="unreachable", severity="warning")

            assert "lark_webhook_send_failed" in caplog.text

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(
        self, notifier: LarkWebhookNotifier,
    ) -> None:
        ok_response = httpx.Response(status_code=200, request=httpx.Request("POST", "https://example.com"))
        err_response = httpx.Response(status_code=500, request=httpx.Request("POST", "https://example.com"))

        with patch.object(
            notifier._client, "post",
            new_callable=AsyncMock,
            side_effect=[err_response, ok_response],
        ) as mock_post, patch("miles.utils.ft.platform.lark_notifier.asyncio.sleep", new_callable=AsyncMock):
            await notifier.send(title="Alert", content="retry test", severity="warning")

        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausts_all_attempts(
        self, notifier: LarkWebhookNotifier, caplog: pytest.LogCaptureFixture,
    ) -> None:
        err_response = httpx.Response(status_code=500, request=httpx.Request("POST", "https://example.com"))

        with patch.object(
            notifier._client, "post",
            new_callable=AsyncMock,
            return_value=err_response,
        ) as mock_post, patch(
            "miles.utils.ft.platform.lark_notifier.asyncio.sleep", new_callable=AsyncMock,
        ):
            with caplog.at_level(logging.ERROR), pytest.raises(httpx.HTTPStatusError):
                await notifier.send(title="Alert", content="fail all", severity="critical")

        assert mock_post.call_count == 3
        assert "lark_webhook_send_failed_all_retries" in caplog.text

    @pytest.mark.asyncio
    async def test_retry_uses_exponential_backoff(
        self, notifier: LarkWebhookNotifier,
    ) -> None:
        err = httpx.ConnectError("refused")

        with patch.object(
            notifier._client, "post",
            new_callable=AsyncMock,
            side_effect=err,
        ), patch(
            "miles.utils.ft.platform.lark_notifier.asyncio.sleep", new_callable=AsyncMock,
        ) as mock_sleep, pytest.raises(httpx.ConnectError):
            await notifier.send(title="Alert", content="backoff", severity="critical")

        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)
