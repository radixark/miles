from typing import Any

from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.ray.rollout.test_external_engine_provider import _make_args, _regular_payload, fake_api_client

from miles.ray.rollout.external_engine_provider import StaticInferenceEngineWorkerProvider

__all__ = ["fake_api_client"]


class TestResolvedExternalAuth:
    async def test_discovery_and_published_metadata_use_the_resolved_api_key(self, fake_api_client: Any) -> None:
        """The actual external provider authenticates SDK probes and publishes the same resolved key."""
        args = _make_args(["host1:8000"])
        args.sglang = make_sglang_config(api_key="resolved-secret")
        vars(args).pop("sglang_api_key", None)
        fake_api_client.answers = [_regular_payload()]
        provider = StaticInferenceEngineWorkerProvider(args=args)

        await provider.init()

        assert fake_api_client.calls == ["http://host1:8000"]
        assert fake_api_client.api_keys == ["resolved-secret"]
        [cell] = provider.cell_infos
        assert cell.meta["sglang_api_key"] == "resolved-secret"
