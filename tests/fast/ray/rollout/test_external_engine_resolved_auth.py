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

    async def test_admin_api_key_and_env_unlock_external_admin_endpoints(
        self, fake_api_client: Any, monkeypatch: Any
    ) -> None:
        """Admin API key (--sglang-admin-api-key or SGLANG_ADMIN_API_KEY) authenticates management probes."""
        monkeypatch.setenv("SGLANG_ADMIN_API_KEY", "env-admin-secret")
        args = _make_args(["host1:8000"])
        args.sglang = make_sglang_config(api_key=None)
        fake_api_client.answers = [_regular_payload()]
        provider = StaticInferenceEngineWorkerProvider(args=args)

        await provider.init()

        assert fake_api_client.api_keys == ["env-admin-secret"]
        [cell] = provider.cell_infos
        assert cell.meta["sglang_api_key"] == "env-admin-secret"

        args_explicit = _make_args(["host1:8000"])
        args_explicit.sglang = make_sglang_config(api_key="normal-secret", admin_api_key="explicit-admin-secret")
        fake_api_client.answers = [_regular_payload()]
        provider_explicit = StaticInferenceEngineWorkerProvider(args=args_explicit)

        await provider_explicit.init()

        assert fake_api_client.api_keys[-1] == "explicit-admin-secret"
        [cell_explicit] = provider_explicit.cell_infos
        assert cell_explicit.meta["sglang_api_key"] == "explicit-admin-secret"
