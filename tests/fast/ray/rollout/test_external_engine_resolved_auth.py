from typing import Any

import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.ray.rollout.test_external_engine_provider import _make_args, _regular_payload, fake_api_client

from miles.ray.rollout.external_engine_provider import StaticInferenceEngineWorkerProvider

__all__ = ["fake_api_client"]


class TestResolvedExternalAuth:
    async def test_discovery_and_published_metadata_use_the_resolved_keys(self, fake_api_client: Any) -> None:
        """The actual external provider authenticates SDK probes and publishes the same resolved keys."""
        args = _make_args(["host1:8000"])
        args.sglang = make_sglang_config(api_key="resolved-secret", admin_api_key="resolved-admin-secret")
        vars(args).pop("sglang_api_key", None)
        fake_api_client.answers = [_regular_payload()]
        provider = StaticInferenceEngineWorkerProvider(args=args)

        await provider.init()

        assert fake_api_client.calls == ["http://host1:8000"]
        assert fake_api_client.api_keys == ["resolved-secret"]
        [cell] = provider.cell_infos
        assert cell.meta["sglang_api_key"] == "resolved-secret"
        assert cell.meta["sglang_admin_api_key"] == "resolved-admin-secret"

    @pytest.mark.parametrize(
        ("flag", "env", "expected"),
        [
            pytest.param("from-flag", "from-env", "from-flag", id="flag-wins"),
            pytest.param(None, "from-env", "from-env", id="env-fallback"),
            pytest.param("", "from-env", "", id="empty-flag-means-none"),
            pytest.param(None, None, None, id="no-key"),
        ],
    )
    async def test_the_admin_key_is_the_flag_else_the_environment(
        self, fake_api_client: Any, monkeypatch: pytest.MonkeyPatch, flag, env, expected
    ) -> None:
        """For engines miles does not launch, like SGLang's own admin clients; the flag in args stays as given."""
        if env is None:
            monkeypatch.delenv("SGLANG_ADMIN_API_KEY", raising=False)
        else:
            monkeypatch.setenv("SGLANG_ADMIN_API_KEY", env)
        args = _make_args(["host1:8000"])
        args.sglang = make_sglang_config(api_key="normal-secret", admin_api_key=flag)
        fake_api_client.answers = [_regular_payload()]
        provider = StaticInferenceEngineWorkerProvider(args=args)

        await provider.init()

        [cell] = provider.cell_infos
        assert cell.meta["sglang_admin_api_key"] == expected
        assert args.sglang.base_args["admin_api_key"] == flag
