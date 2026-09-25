from collections.abc import Iterator
from typing import Any

import pytest
from transformers import AutoConfig, PretrainedConfig


@pytest.fixture
def policy_hf_configs(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, dict[str, Any]]]:
    configs: dict[str, dict[str, Any]] = {}

    def load(checkpoint: str, **kwargs: Any) -> PretrainedConfig:
        return PretrainedConfig(**configs.get(checkpoint, {}))

    monkeypatch.setattr(AutoConfig, "from_pretrained", load)
    yield configs
