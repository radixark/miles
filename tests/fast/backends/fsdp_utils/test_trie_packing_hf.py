"""The FSDP actor's trie path must score every sample of a tiny HF model as if it ran alone.

Every kernel is replaced by a plain-PyTorch reference, so this checks everything around the kernels on CPU: the
registered attention and mask functions, the actor's model inputs, logical positions through HF RoPE, sliding
windows, and response-only logits. ``tests/fast-gpu/test_fsdp_trie_packing.py`` runs the kernels.
"""

import random
import sys
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.reference_trie_attention import reference_trie_attention
from transformers import Glm4MoeLiteConfig, Qwen3Config, Qwen3MoeConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from miles.backends.fsdp_utils import actor as fsdp_actor
from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.backends.fsdp_utils.adaptations.specs import resolve_arch_adapter
from miles.backends.fsdp_utils.adaptations.trie_attention import (
    register_trie_attention,
    trie_attn_implementation,
    trie_model_spec,
)
from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.data import trie_batch
from miles.backends.training_utils.data.trie_batch import TrieMicroBatches
from miles.backends.training_utils.data.trie_packing import TrieKV, TrieModelSpec

_VOCAB = 128
_WINDOW = 12


def _reference_flash_attn_varlen(
    q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, window_size, softmax_scale
):
    """flash_attn_varlen_func's contract, bottom-right causal, on the trie reference."""
    assert causal
    kv = TrieKV(index=torch.arange(k.size(0)), cu_seqlens=cu_seqlens_k, max_seqlen=max_seqlen_k)
    window_left = None if window_size[0] < 0 else window_size[0]
    return reference_trie_attention(q, k, v, kv, cu_seqlens_q, window_left=window_left, softmax_scale=softmax_scale)


@pytest.fixture
def cpu_trie(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=1), tp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    monkeypatch.setitem(
        sys.modules, "flash_attn", SimpleNamespace(flash_attn_varlen_func=_reference_flash_attn_varlen)
    )
    # a key tile short enough that these prompts' window gathers drop ancestors
    monkeypatch.setattr(trie_batch, "WINDOW_START_MULTIPLE", 4)
    register_trie_attention("flash_attention_2")


def _actor_for(model):
    actor = object.__new__(FSDPTrainRayActor)
    actor.arch_adapter = resolve_arch_adapter(model.config)
    return actor


def _samples(seed: int) -> dict:
    """Two prompts of three samples sharing a system prompt longer than the window; two of each prompt's
    responses open alike, so they share rows too."""
    rng = random.Random(seed)
    system = [rng.randrange(1, _VOCAB) for _ in range(20)]
    tokens, total_lengths, response_lengths = [], [], []
    for _ in range(2):
        prompt = system + [rng.randrange(1, _VOCAB) for _ in range(rng.randrange(4, 10))]
        opening = [rng.randrange(1, _VOCAB) for _ in range(2)]
        for sample in range(3):
            response = (opening if sample < 2 else []) + [rng.randrange(1, _VOCAB) for _ in range(rng.randrange(3, 8))]
            tokens.append(torch.tensor(prompt + response))
            total_lengths.append(len(prompt) + len(response))
            response_lengths.append(len(response))
    return {
        "tokens": tokens,
        "loss_masks": [torch.ones(r, dtype=torch.int) for r in response_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }


def _tiny_model(kind: str, sliding: bool):
    torch.manual_seed(0)
    common = dict(
        vocab_size=_VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=_WINDOW if sliding else None,
        use_sliding_window=sliding,
        layer_types=["sliding_attention", "full_attention"] if sliding else ["full_attention"] * 2,
    )
    if kind == "qwen3":
        return Qwen3ForCausalLM._from_config(Qwen3Config(intermediate_size=128, **common), attn_implementation="sdpa")
    if kind == "glm4_moe_lite":
        # multi-head latent attention: values narrower than queries and keys
        config = Glm4MoeLiteConfig(
            vocab_size=_VOCAB,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            q_lora_rank=None,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=8,
        )
        return Glm4MoeLiteForCausalLM._from_config(config, attn_implementation="sdpa")
    config = Qwen3MoeConfig(moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2, **common)
    return Qwen3MoeForCausalLM._from_config(config, attn_implementation="sdpa")


def _log_probs(logits, targets):
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _solo(model, samples) -> torch.Tensor:
    model.set_attn_implementation("sdpa")
    out = []
    for index, total in enumerate(samples["total_lengths"]):
        tokens, response = samples["tokens"][index], samples["response_lengths"][index]
        positions = torch.arange(total)[None]
        logits = model(
            input_ids=tokens[None],
            position_ids=positions,
            use_cache=False,
        ).logits
        out.append(_log_probs(logits[0, total - response - 1 : total - 1], tokens[-response:]))
    return torch.cat(out)


def _trie_iterator(rollout, model_spec: TrieModelSpec, micro_batch_indices=None) -> data_utils.DataIterator:
    micro_batch_indices = micro_batch_indices or [list(range(len(rollout["tokens"])))]
    trie_micro_batches = TrieMicroBatches(rollout, model_spec, micro_batch_indices)
    return data_utils.DataIterator(
        rollout, micro_batch_indices=micro_batch_indices, trie_micro_batches=trie_micro_batches
    )


def _trie(model, samples) -> torch.Tensor:
    model.set_attn_implementation(trie_attn_implementation(model.config, "sdpa"))
    keys = ["tokens", "loss_masks", "total_lengths", "response_lengths"]
    model_spec = trie_model_spec(model.config.get_text_config())
    batch = data_utils.get_batch(
        _trie_iterator(samples, model_spec),
        keys,
        pad_multiplier=8,
        qkv_format="thd",
    )
    layout = batch["trie_layout"]
    assert layout.num_tokens < sum(samples["total_lengths"])
    if layout.window_kv is not None:
        assert layout.window_kv.index.numel() < layout.kv.index.numel(), "no window gather dropped an ancestor"
    model_args = _actor_for(model)._get_model_inputs_args(batch)
    logits = fsdp_actor._logits(model, model_args, batch)[0]
    targets = torch.cat([t[-r:] for t, r in zip(batch["unconcat_tokens"], batch["response_lengths"], strict=True)])
    return _log_probs(logits, targets)


@pytest.mark.parametrize(
    ("kind", "sliding"),
    [
        ("qwen3", False),
        ("qwen3", True),
        ("qwen3_moe", False),
        ("qwen3_moe", True),
        ("glm4_moe_lite", False),
    ],
    ids=[
        "qwen3-full",
        "qwen3-sliding",
        "qwen3_moe-full",
        "qwen3_moe-sliding",
        "glm4_moe_lite-mla",
    ],
)
def test_trie_row_scores_and_trains_each_sample_as_if_alone(cpu_trie, kind, sliding):
    model = _tiny_model(kind, sliding)
    samples = _samples(0)

    model.zero_grad()
    solo = _solo(model, samples)
    solo.sum().backward()
    solo_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    trie = _trie(model, samples)
    trie.sum().backward()

    torch.testing.assert_close(trie, solo, rtol=0, atol=1e-5)
    for name, p in model.named_parameters():
        if name in solo_grads:
            torch.testing.assert_close(p.grad, solo_grads[name], rtol=1e-4, atol=1e-6, msg=name)
