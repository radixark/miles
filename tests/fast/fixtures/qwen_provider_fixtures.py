from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.backends.megatron_utils import model_provider
from miles.backends.training_utils import parallel
from miles.utils.args.trainer_utils import compute_trainer_config


@pytest.fixture
def qwen_provider_sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    parsed = parse_megatron_test_config(
        "--custom-model-provider-path",
        "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider",
        "--megatron-to-hf-mode",
        "raw",
        "--transformer-impl",
        "local",
        "--vocab-size",
        "8",
        "--make-vocab-size-divisible-by",
        "1",
    )
    args = compute_trainer_config(all_config=parsed, trainer=parsed.raw_megatron.trainers[0])
    constructed: list[dict[str, Any]] = []
    config = SimpleNamespace(use_kitchen=False, use_kitchen_attention=False, kitchen_attention_backend=None)

    def build_model(**kwargs: Any) -> torch.nn.Module:
        constructed.append(kwargs)
        result = torch.nn.Module()
        result.config = kwargs["config"]
        return result

    rank = SimpleNamespace(rank=0)
    monkeypatch.setattr(parallel, "_parallel_state", SimpleNamespace(pp=rank, tp=rank, cp=rank, intra_dp=rank))
    monkeypatch.setattr(model_provider, "core_transformer_config_from_args", lambda _args: config)
    monkeypatch.setattr(model_provider, "get_gpt_layer_local_spec", lambda **kwargs: object())
    monkeypatch.setattr(model_provider, "GPTModel", build_model)
    return SimpleNamespace(args=args, constructed=constructed)
