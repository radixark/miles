"""The entry point examples/multi_lora/harbor_tinker/run_harbor_tinker.py: config → cookbook train.Config, and the sandbox preflight.

Skipped when tinker-cookbook is not installed. The preflight reuses ``miles.rollout.agentic.credentials``; the tests only
check what the entry point adds on top: the env-var contract and the AgentENV reachability probe.
"""

import pytest

pytest.importorskip("tinker_cookbook")
from examples.multi_lora.harbor_tinker import run_harbor_tinker  # noqa: E402
from examples.multi_lora.harbor_tinker.harbor_env import HarborDatasetBuilder, SessionRolloutStrategy  # noqa: E402
from examples.multi_lora.harbor_tinker.run_harbor_tinker import (  # noqa: E402
    HarborTinkerConfig,
    build_config,
    preflight_sandbox,
)


def _config(**overrides) -> HarborTinkerConfig:
    fields = dict(
        gateway="http://gateway:10613", model_name="Qwen/Qwen3-30B-A3B", tasks_dir="/tasks", api_key="tml-key"
    )
    fields.update(overrides)
    return HarborTinkerConfig(**fields)


def test_build_config_is_a_cookbook_config_with_our_plug_ins():
    """train.Config gets the gateway as base_url, our dataset builder and our rollout strategy; everything else is cookbook defaults."""
    config = build_config(_config(groups_per_batch=2, group_size=3, max_tokens=512, lora_rank=8, max_steps=7))
    assert config.base_url == "http://gateway:10613" and config.model_name == "Qwen/Qwen3-30B-A3B"
    assert config.recipe_name == "harbor-tinker" and config.loss_fn == "ppo" and config.max_steps == 7
    assert config.lora_rank == 8 and config.max_tokens == 512 and config.learning_rate == 3e-5

    builder = config.dataset_builder
    assert isinstance(builder, HarborDatasetBuilder)
    assert (builder.tasks_dir, builder.groups_per_batch, builder.group_size) == ("/tasks", 2, 3)

    strategy = config.rollout_error_tolerance
    assert isinstance(strategy, SessionRolloutStrategy)
    assert (strategy.gateway_url, strategy.api_key, strategy.max_tokens) == ("http://gateway:10613", "tml-key", 512)
    assert config.effective_rollout_strategy() is strategy


def test_build_config_needs_the_tenant_key(monkeypatch):
    """The Tinker key comes from the config or TINKER_API_KEY; without either there is no tenant to bind sessions for."""
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        build_config(_config(api_key=None))
    monkeypatch.setenv("TINKER_API_KEY", "tml-from-env")
    assert build_config(_config(api_key=None)).rollout_error_tolerance.api_key == "tml-from-env"


def test_preflight_checks_env_credentials_sdk_and_endpoint(monkeypatch, tmp_path):
    """Each missing piece fails with a message naming it: HARBOR_ENV_TYPE, HARBOR_TASKS_DIR, the provider, its key file, the SDK, the endpoint."""
    for var in ("HARBOR_ENV_TYPE", "HARBOR_TASKS_DIR", "E2B_API_KEY", "E2B_API_KEY_FILE", "E2B_API_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="HARBOR_ENV_TYPE"):
        preflight_sandbox()

    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    with pytest.raises(RuntimeError, match="HARBOR_TASKS_DIR"):
        preflight_sandbox()

    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "nosuchbox")
    with pytest.raises(RuntimeError, match="nosuchbox"):
        preflight_sandbox()

    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    monkeypatch.setenv("E2B_API_KEY_FILE", str(tmp_path / "missing-key"))
    with pytest.raises(RuntimeError, match="credential"):
        preflight_sandbox()

    key_file = tmp_path / "e2b_key"
    key_file.write_text("agentenv-key\n")
    monkeypatch.setenv("E2B_API_KEY_FILE", str(key_file))
    monkeypatch.setattr(
        run_harbor_tinker, "preflight_sdk", lambda *args, **kwargs: None
    )  # the SDK check is credentials.py's
    preflight_sandbox()  # E2B Cloud (no E2B_API_URL): nothing to probe

    monkeypatch.setenv(
        "E2B_API_URL", "http://127.0.0.1:9"
    )  # nothing listens here: the cluster is stopped or we are off the tailnet
    with pytest.raises(RuntimeError, match="E2B_API_URL"):
        preflight_sandbox()
