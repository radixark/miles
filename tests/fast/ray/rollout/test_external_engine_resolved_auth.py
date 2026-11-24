import argparse
from typing import Any

from sglang.srt.server_args import ServerArgs
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.ray.rollout.test_external_engine_provider import _make_args, _regular_payload, fake_api_client

from miles.backends.sglang_utils.arguments import _add_prefixed_server_args
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
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
        assert "sglang_admin_api_key" not in cell.meta

    async def test_admin_api_key_and_dual_key_servers_route_normal_vs_admin_headers(
        self, fake_api_client: Any, monkeypatch: Any
    ) -> None:
        """Admin-only servers use admin_api_key everywhere; dual-key servers keep api_key for NORMAL routes."""
        monkeypatch.setenv("SGLANG_ADMIN_API_KEY", "env-admin-secret")
        args = _make_args(["host1:8000"])
        args.sglang = make_sglang_config(api_key=None)
        fake_api_client.answers = [_regular_payload()]
        provider = StaticInferenceEngineWorkerProvider(args=args)

        await provider.init()

        assert fake_api_client.api_keys == ["env-admin-secret"]
        [cell] = provider.cell_infos
        assert cell.meta["sglang_api_key"] == "env-admin-secret"
        assert cell.meta["sglang_admin_api_key"] == "env-admin-secret"

        args_dual = _make_args(["host1:8000"])
        args_dual.sglang = make_sglang_config(api_key="normal-secret", admin_api_key="explicit-admin-secret")
        fake_api_client.answers = [_regular_payload()]
        provider_dual = StaticInferenceEngineWorkerProvider(args=args_dual)

        await provider_dual.init()

        assert fake_api_client.api_keys[-1] == "normal-secret"
        [cell_dual] = provider_dual.cell_infos
        assert cell_dual.meta["sglang_api_key"] == "normal-secret"
        assert cell_dual.meta["sglang_admin_api_key"] == "explicit-admin-secret"

        client = SGLangApiClient(
            server_url="http://host1:8000",
            api_key=cell_dual.meta["sglang_api_key"],
            admin_api_key=cell_dual.meta["sglang_admin_api_key"],
        )
        assert client._normal_headers["Authorization"] == "Bearer normal-secret"
        assert client._headers["Authorization"] == "Bearer explicit-admin-secret"


def test_deprecated_choice_to_flag_action_prefixes_targets_without_duplicate_default(monkeypatch: Any) -> None:
    class DeprecatedChoiceToFlagAction(argparse.Action):
        def __init__(self, option_strings, dest, *, flags: dict[str, tuple[str, str]], **kwargs):
            if "default" in kwargs:
                raise TypeError("DeprecatedChoiceToFlagAction sets its own default=SUPPRESS")
            self.flags = flags
            super().__init__(option_strings, dest, default=argparse.SUPPRESS, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            target_dest, _new_flag = self.flags[values]
            setattr(namespace, target_dest, True)

    def _fake_add_cli_args(parser):
        parser.add_argument("--enable-foo", action="store_true")
        parser.add_argument(
            "--legacy-mode",
            action=DeprecatedChoiceToFlagAction,
            choices=["foo"],
            flags={"foo": ("enable_foo", "--enable-foo")},
        )

    monkeypatch.setattr(ServerArgs, "add_cli_args", staticmethod(_fake_add_cli_args))
    parser = argparse.ArgumentParser()
    _add_prefixed_server_args(
        parser, flag_prefix="eval-sglang", dest_prefix="eval_sglang_", skipped_args=[], inherit=True
    )
    parsed = parser.parse_args(["--eval-sglang-legacy-mode", "foo"])
    assert vars(parsed) == {"eval_sglang_enable_foo": True}
