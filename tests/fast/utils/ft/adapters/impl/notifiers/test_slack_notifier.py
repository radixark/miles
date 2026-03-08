from unittest.mock import AsyncMock, patch

import pytest
from tests.fast.utils.ft.platform.notifiers.conftest import make_ok_response

from miles.utils.ft.adapters.impl.notifiers.slack_notifier import SlackWebhookNotifier


class TestSlackWebhookNotifierPayload:
    @pytest.fixture
    async def notifier(self) -> SlackWebhookNotifier:
        instance = SlackWebhookNotifier(
            webhook_url="https://hooks.slack.com/services/T00/B00/XXXX",
        )
        yield instance
        await instance.aclose()

    @pytest.mark.anyio
    async def test_send_posts_block_kit_payload(self, notifier: SlackWebhookNotifier) -> None:
        with patch.object(
            notifier._client, "post", new_callable=AsyncMock, return_value=make_ok_response()
        ) as mock_post:
            await notifier.send(title="GPU OOM", content="Node-3 ran out of memory", severity="critical")

            mock_post.assert_called_once()
            payload = mock_post.call_args[1]["json"]

            assert "blocks" in payload
            blocks = payload["blocks"]
            assert len(blocks) == 2

            header = blocks[0]
            assert header["type"] == "header"
            assert "[critical]" in header["text"]["text"]
            assert "GPU OOM" in header["text"]["text"]

            section = blocks[1]
            assert section["type"] == "section"
            assert section["text"]["type"] == "mrkdwn"
            assert section["text"]["text"] == "Node-3 ran out of memory"

    @pytest.mark.anyio
    @pytest.mark.parametrize("severity", ["critical", "warning", "info"])
    async def test_send_includes_severity_in_header(
        self,
        notifier: SlackWebhookNotifier,
        severity: str,
    ) -> None:
        with patch.object(
            notifier._client, "post", new_callable=AsyncMock, return_value=make_ok_response()
        ) as mock_post:
            await notifier.send(title="Alert", content="test", severity=severity)

            payload = mock_post.call_args[1]["json"]
            header_text = payload["blocks"][0]["text"]["text"]
            assert f"[{severity}]" in header_text

    @pytest.mark.anyio
    async def test_unknown_severity_uses_default_emoji(self, notifier: SlackWebhookNotifier) -> None:
        with patch.object(
            notifier._client, "post", new_callable=AsyncMock, return_value=make_ok_response()
        ) as mock_post:
            await notifier.send(title="Alert", content="test", severity="unknown")

            payload = mock_post.call_args[1]["json"]
            header_text = payload["blocks"][0]["text"]["text"]
            assert ":white_circle:" in header_text
