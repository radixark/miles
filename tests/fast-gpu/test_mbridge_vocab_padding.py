"""mbridge loads an HF embedding into a miles-built model at that model's vocab rows."""

from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    suite="stage-b-2-gpu-h200",
    labels=["ckpt", "lora", "miles-plugin"],
    hardware=["hopper", "blackwell"],
    num_gpus=1,
)

import transformers
from mbridge.core.parallel_states import ParallelStates

from miles.backends.megatron_utils.checkpoint import create_mbridge_for_model

HF_VOCAB = 10


def _save_config(path, **vocab):
    transformers.Qwen3Config(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        **vocab,
    ).save_pretrained(path)
    return str(path)


@pytest.fixture
def hf_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ParallelStates, "get_parallel_state", classmethod(lambda cls: cls()))
    return _save_config(tmp_path, vocab_size=HF_VOCAB)


@pytest.fixture
def padded_hf_dir(tmp_path, monkeypatch):
    """Like Inkling: vocab_size is the unpadded vocab, the embedding holds padded_vocab_size rows."""
    monkeypatch.setattr(ParallelStates, "get_parallel_state", classmethod(lambda cls: cls()))
    return _save_config(tmp_path, vocab_size=HF_VOCAB - 2, padded_vocab_size=HF_VOCAB)


@pytest.mark.parametrize("model_vocab_rows", [12, HF_VOCAB], ids=["megatron-padded", "built-unpadded"])
def test_the_hf_embedding_loads_padded_with_zeros_to_the_built_models_rows(hf_dir, model_vocab_rows):
    """Unpadded rows fail the TP scatter of a padded model."""
    bridge = create_mbridge_for_model([SimpleNamespace(vocab_size=model_vocab_rows)], hf_dir)

    rows = bridge._weight_to_mcore_format("embedding.word_embeddings.weight", [torch.ones(HF_VOCAB, 4)])

    assert rows.shape == (model_vocab_rows, 4)
    assert rows[:HF_VOCAB].eq(1).all() and not rows[HF_VOCAB:].any()


@pytest.mark.parametrize("model_vocab_rows", [12, HF_VOCAB], ids=["megatron-padded", "built-unpadded"])
def test_a_checkpoint_with_a_padded_embedding_loads_its_stored_rows(padded_hf_dir, model_vocab_rows):
    bridge = create_mbridge_for_model([SimpleNamespace(vocab_size=model_vocab_rows)], padded_hf_dir)

    rows = bridge._weight_to_mcore_format("embedding.word_embeddings.weight", [torch.ones(HF_VOCAB, 4)])

    assert rows.shape == (model_vocab_rows, 4)
    assert rows[:HF_VOCAB].eq(1).all() and not rows[HF_VOCAB:].any()


def test_a_model_built_with_fewer_rows_than_the_hf_vocab_is_refused(hf_dir):
    with pytest.raises(AssertionError, match="fewer vocab rows"):
        create_mbridge_for_model([SimpleNamespace(vocab_size=HF_VOCAB - 1)], hf_dir)
