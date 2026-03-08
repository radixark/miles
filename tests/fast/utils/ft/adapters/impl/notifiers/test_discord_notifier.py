from unittest.mock import AsyncMock, patch

import pytest
from tests.fast.utils.ft.platform.notifiers.conftest import make_ok_response

from miles.utils.ft.adapters.impl.notifiers.discord_notifier import DiscordWebhookNotifier


class TestDiscordWebhookNotifierPayload:
    @pytest.fixture
    async def notifier(self) -> DiscordWebhookNotifier:
        instance = DiscordWebhookNotifier(
            webhook_url="https://discord.com/api/webhooks/123/abc",
        )
        yield instance
        await instance.aclose()

    @pytest.mark.anyio
    async def test_send_posts_embed_payload(self, notifier: DiscordWebhookNotifier) -> None:
        with patch.object(
            notifier._client, "post", new_callable=AsyncMock, return_value=make_ok_response()
        ) as mock_post:
            await notifier.send(title="GPU OOM", content="Node-3 ran out of memory", severity="critical")

            mock_post.assert_called_once()
            payload = mock_post.call_args[1]["json"]

            assert "embeds" in payload
            embeds = payload["embeds"]
            assert len(embeds) == 1

            embed = embeds[0]
            assert embed["title"] == "[critical] GPU OOM"
            assert embed["description"] == "Node-3 ran out of memory"
            assert embed["color"] == 0xE74C3C

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("severity", "expected_color"),
        [("critical", 0xE74C3C), ("warning", 0xF1C40F), ("info", 0x3498DB)],
    )
    async def test_severity_color_mapping(
        self,
        notifier: DiscordWebhookNotifier,
        severity: str,
        expected_color: int,
    ) -> None:
        with patch.object(
            notifier._client, "post", new_callable=AsyncMock, return_value=make_ok_response()
        ) as mock_post:
            await notifier.send(title="Alert", content="test", severity=severity)

            payload = mock_post.call_args[1]["json"]
            assert payload["embeds"][0]["color"] == expected_color

    @pytest.mark.anyio
    async def test_unknown_severity_uses_default_color(self, notifier: DiscordWebhookNotifier) -> None:
        with patch.object(
            notifier._client, "post", new_callable=AsyncMock, return_value=make_ok_response()
        ) as mock_post:
            await notifier.send(title="Alert", content="test", severity="unknown")

            payload = mock_post.call_args[1]["json"]
            assert payload["embeds"][0]["color"] == 0x95A5A6
