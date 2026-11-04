from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from tests.fast.fixtures.args_fixtures import ConfigNamespace, make_trainer_args
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from miles.backends.fsdp_utils.arguments import FSDPArgs
from miles.backends.training_utils.parallel import ParallelState
from miles.utils import distributed_utils


class _OptimizerReached(RuntimeError):
    pass


@dataclass
class _InitBackendCapture:
    checkpoint: Path
    optimizers: list[torch.optim.Optimizer] = field(default_factory=list)
    shard_options: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def init_backend_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fsdp_three_rank_state: ParallelState
) -> _InitBackendCapture:
    checkpoint = tmp_path / "tiny-gpt2"
    checkpoint.mkdir()
    config = GPT2Config(vocab_size=4, n_positions=8, n_embd=8, n_layer=1, n_head=2, tie_word_embeddings=True)
    GPT2LMHeadModel(config).save_pretrained(checkpoint)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]")))
    tokenizer.save_pretrained(checkpoint)
    capture = _InitBackendCapture(checkpoint=checkpoint)
    original_adamw: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW

    def capture_adamw(*args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        capture.optimizers.append(original_adamw(*args, **kwargs))
        raise _OptimizerReached("Optimizer constructed from backend configuration")

    def fully_shard(model: torch.nn.Module, **kwargs: Any) -> None:
        capture.shard_options.append(kwargs)

    def set_model_state_dict(model: torch.nn.Module, state: dict, **kwargs: Any) -> None:
        model.load_state_dict(state)

    fsdp_three_rank_state.meshes["fsdp"] = None
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(dist, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", object())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", fully_shard)
    monkeypatch.setattr("torch.distributed.checkpoint.state_dict.set_model_state_dict", set_model_state_dict)
    monkeypatch.setattr(torch.optim, "AdamW", capture_adamw)
    return capture


def _make_fsdp_init_args(**overrides: Any) -> ConfigNamespace:
    return make_trainer_args(**(vars(FSDPArgs()) | {"train_backend": "fsdp"} | overrides))
