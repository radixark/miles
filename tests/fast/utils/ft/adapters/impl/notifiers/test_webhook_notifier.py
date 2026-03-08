import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from tests.fast.utils.ft.adapters.impl.notifiers.conftest import make_error_response, make_ok_response

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


# ---------------------------------------------------------------------------
# Real HTTP server tests — exercises actual HTTP POST / retry / timeout
# ---------------------------------------------------------------------------


class TestWebhookNotifierRealHttp:
    """Integration tests using a real local HTTP server (aiohttp.web)."""

    @pytest.mark.anyio
    async def test_send_posts_to_real_http_server(self) -> None:
        """A real HTTP POST should arrive at a local server with the expected payload."""
        import asyncio
        import json

        from aiohttp import web

        received_payloads: list[dict] = []

        async def handler(request: web.Request) -> web.Response:
            body = await request.json()
            received_payloads.append(body)
            return web.Response(status=200)

        app = web.Application()
        app.router.add_post("/hook", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()

        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/hook"

        try:
            notifier = LarkWebhookNotifier(webhook_url=url)
            try:
                await notifier.send(title="Test Alert", content="hello world", severity="warning")
            finally:
                await notifier.aclose()

            assert len(received_payloads) == 1
            payload = received_payloads[0]
            assert payload["msg_type"] == "interactive"
            assert "Test Alert" in json.dumps(payload)
        finally:
            await runner.cleanup()

    @pytest.mark.anyio
    async def test_retry_on_server_error(self) -> None:
        """Server returns 500 twice then 200 — notifier should succeed via retry."""
        from aiohttp import web

        call_count = 0

        async def handler(request: web.Request) -> web.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return web.Response(status=500, text="Internal Server Error")
            return web.Response(status=200)

        app = web.Application()
        app.router.add_post("/hook", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()

        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/hook"

        try:
            notifier = LarkWebhookNotifier(webhook_url=url)
            try:
                await notifier.send(title="Retry Test", content="should retry", severity="critical")
            finally:
                await notifier.aclose()

            assert call_count == 3
        finally:
            await runner.cleanup()

    @pytest.mark.anyio
    async def test_timeout_on_slow_server(self) -> None:
        """Server delays response longer than httpx timeout — should raise."""
        import asyncio

        from aiohttp import web

        async def handler(request: web.Request) -> web.Response:
            await asyncio.sleep(30)
            return web.Response(status=200)

        app = web.Application()
        app.router.add_post("/hook", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()

        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/hook"

        try:
            notifier = LarkWebhookNotifier(webhook_url=url)
            notifier._client.timeout = httpx.Timeout(0.5)
            try:
                with pytest.raises((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException)):
                    await notifier.send(title="Slow", content="timeout test", severity="warning")
            finally:
                await notifier.aclose()
        finally:
            await runner.cleanup()
