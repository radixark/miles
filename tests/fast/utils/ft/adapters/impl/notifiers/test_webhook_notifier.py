import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from tests.fast.utils.ft.platform.notifiers.conftest import make_error_response, make_ok_response

from miles.utils.ft.adapters.impl.notifiers.lark_notifier import LarkWebhookNotifier

_SLEEP_PATCH = "miles.utils.ft.utils.retry.asyncio.sleep"


class TestWebhookNotifierRetry:
    """Tests for the retry/backoff logic in the BaseWebhookNotifier base class.

    Uses LarkWebhookNotifier as the concrete implementation.
    """

    @pytest.fixture
    async def notifier(self) -> LarkWebhookNotifier:
        instance = LarkWebhookNotifier(
            webhook_url="https://open.larksuite.com/open-apis/bot/v2/hook/test-token",
        )
        yield instance
        await instance.aclose()

    @pytest.mark.anyio
    async def test_http_error_raises_after_retries(
        self,
        notifier: LarkWebhookNotifier,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch.object(notifier._client, "post", new_callable=AsyncMock, return_value=make_error_response()), patch(
            _SLEEP_PATCH, new_callable=AsyncMock
        ):
            with caplog.at_level(logging.WARNING), pytest.raises(httpx.HTTPStatusError):
                await notifier.send(title="Fault Alert", content="test error", severity="critical")

            assert "retry_failed" in caplog.text

    @pytest.mark.anyio
    async def test_connect_error_raises_after_retries(
        self,
        notifier: LarkWebhookNotifier,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch.object(
            notifier._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ), patch(_SLEEP_PATCH, new_callable=AsyncMock):
            with caplog.at_level(logging.WARNING), pytest.raises(httpx.ConnectError):
                await notifier.send(title="Alert", content="unreachable", severity="warning")

            assert "retry_failed" in caplog.text

    @pytest.mark.anyio
    async def test_retry_succeeds_on_second_attempt(
        self,
        notifier: LarkWebhookNotifier,
    ) -> None:
        with patch.object(
            notifier._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[make_error_response(), make_ok_response()],
        ) as mock_post, patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await notifier.send(title="Alert", content="retry test", severity="warning")

        assert mock_post.call_count == 2

    @pytest.mark.anyio
    async def test_retry_exhausts_all_attempts(
        self,
        notifier: LarkWebhookNotifier,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch.object(
            notifier._client,
            "post",
            new_callable=AsyncMock,
            return_value=make_error_response(),
        ) as mock_post, patch(_SLEEP_PATCH, new_callable=AsyncMock):
            with caplog.at_level(logging.ERROR), pytest.raises(httpx.HTTPStatusError):
                await notifier.send(title="Alert", content="fail all", severity="critical")

        assert mock_post.call_count == 3
        assert "retry_exhausted" in caplog.text

    @pytest.mark.anyio
    async def test_retry_uses_exponential_backoff(
        self,
        notifier: LarkWebhookNotifier,
    ) -> None:
        err = httpx.ConnectError("refused")

        with patch.object(
            notifier._client,
            "post",
            new_callable=AsyncMock,
            side_effect=err,
        ), patch(
            _SLEEP_PATCH, new_callable=AsyncMock
        ) as mock_sleep, pytest.raises(httpx.ConnectError):
            await notifier.send(title="Alert", content="backoff", severity="critical")

        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)


class TestWebhookNotifierLifecycle:
    @pytest.mark.anyio
    async def test_aclose_is_idempotent(self) -> None:
        notifier = LarkWebhookNotifier(webhook_url="https://example.com/hook/test")
        await notifier.aclose()
        await notifier.aclose()
