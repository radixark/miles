import asyncio
import os
from pathlib import Path

import pytest
from examples.experimental.hud import rollout

from miles.utils.args.custom_view import ImmutableNamespace


class TestHudCredentialOwnership:
    @pytest.mark.parametrize("explicit_file", [False, True])
    def test_declared_key_file_wins_over_environment_fallback_and_does_not_mutate_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_file: bool
    ) -> None:
        """HUD reads declared file configuration while preserving the caller's immutable settings."""
        key = tmp_path / "key"
        key.write_text(" explicit-key \n")
        fallback = tmp_path / "fallback"
        fallback.write_text("environment-key\n")
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
        monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(fallback))
        args = ImmutableNamespace(hud_daytona_key_file=str(key) if explicit_file else None)
        before = dict(args)

        rollout._load_daytona_key(args)

        assert os.environ["DAYTONA_API_KEY"] == ("explicit-key" if explicit_file else "environment-key")
        assert dict(args) == before

    def test_existing_credentials_do_not_require_a_readable_key_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing provider credential takes precedence over a missing configured key file."""
        monkeypatch.setenv("DAYTONA_API_KEY", "already-provisioned")

        rollout._load_daytona_key(ImmutableNamespace(hud_daytona_key_file=str(tmp_path / "missing")))

        assert os.environ["DAYTONA_API_KEY"] == "already-provisioned"

    def test_missing_configured_credentials_fail_before_runtime_creation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing provider credentials fail explicitly without creating a remote runtime."""
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)

        with pytest.raises(ValueError, match="no Daytona credentials"):
            rollout._load_daytona_key(ImmutableNamespace(hud_daytona_key_file=str(tmp_path / "missing")))

        assert "DAYTONA_API_KEY" not in os.environ


class TestHudSandboxLimit:
    async def test_explicit_capacity_is_process_shared_and_releases_waiters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The declared sandbox capacity gates episodes and remains stable for the process lifetime."""
        monkeypatch.setattr(rollout, "_gate", None)
        gate = rollout._sandbox_gate(ImmutableNamespace(hud_max_sandboxes=2))
        assert rollout._sandbox_gate(ImmutableNamespace(hud_max_sandboxes=8)) is gate
        await gate.acquire()
        await gate.acquire()
        pending = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)
        assert not pending.done()

        gate.release()

        assert await asyncio.wait_for(pending, timeout=1) is True
        gate.release()
        gate.release()


class TestHudAgentSettings:
    async def test_declared_agent_settings_reach_the_sdk_and_tool_without_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HUD forwards explicit model, prompt, turn, token and screenshot settings to its SDK boundary."""
        pytest.importorskip("hud")
        from hud.agents.openai_compatible.agent import OpenAIChatAgent

        captured = []
        monkeypatch.setattr(OpenAIChatAgent, "__init__", lambda self, config: captured.append(config))
        args = ImmutableNamespace(
            hud_max_steps=7,
            hud_system_prompt="Limit {max_steps}",
            hud_shot_width=512,
            hud_max_tokens_per_turn=31,
            sglang_router_ip="router",
            sglang_router_port=1234,
            hf_checkpoint="policy",
        )
        sampling = {"temperature": 0.4, "top_p": 0.6}

        agent = rollout._build_agent(args, sampling)

        [config] = captured
        try:
            assert config.model == "policy"
            assert config.max_steps == 7
            assert config.system_prompt == "Limit 7"
            assert config.completion_kwargs == {
                "max_tokens": 31,
                "temperature": 0.4,
                "top_p": 0.6,
                "extra_body": {"return_token_ids": True},
            }
            assert str(config.model_client.base_url) == "http://router:1234/v1/"
            assert agent.tool_catalog[0].shot_width == 512
            assert sampling == {"temperature": 0.4, "top_p": 0.6}
        finally:
            await config.model_client.close()
