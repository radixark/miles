from unittest.mock import AsyncMock, patch

import pytest
from tests.fast.utils.ft.platform.notifiers.conftest import make_ok_response

from miles.utils.ft.adapters.impl.notifiers.lark_notifier import LarkWebhookNotifier


class TestLarkWebhookNotifierPayload:
    @pytest.fixture
    async def notifier(self) -> LarkWebhookNotifier:
        instance = LarkWebhookNotifier(
            webhook_url="https://open.larksuite.com/open-apis/bot/v2/hook/test-token",
        )
        yield instance
        await instance.aclose()

    @pytest.mark.anyio
    async def test_send_posts_correct_json(self, notifier: LarkWebhookNotifier) -> None:
        with patch.object(
            notifier._client, "post", new_callable=AsyncMock, return_value=make_ok_response()
        ) as mock_post:
            await notifier.send(title="Fault Alert", content="GPU lost on node-3", severity="critical")

            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            assert payload["msg_type"] == "interactive"
            assert payload["card"]["header"]["title"]["tag"] == "plain_text"
            assert payload["card"]["header"]["title"]["content"] == "[critical] Fault Alert"
            assert payload["card"]["elements"] == [{"tag": "markdown", "content": "GPU lost on node-3"}]

    @pytest.mark.anyio
    @pytest.mark.parametrize("severity", ["critical", "warning", "info"])
    async def test_send_includes_severity_in_header(
        self,
        notifier: LarkWebhookNotifier,
        severity: str,
    ) -> None:
        with patch.object(
            notifier._client, "post", new_callable=AsyncMock, return_value=make_ok_response()
        ) as mock_post:
            await notifier.send(title="Alert", content="test", severity=severity)

            payload = mock_post.call_args[1]["json"]
            assert payload["card"]["header"]["title"]["content"].startswith(f"[{severity}]")
