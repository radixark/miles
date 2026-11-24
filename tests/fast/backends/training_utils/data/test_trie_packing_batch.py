"""A trie-packed micro-batch must reach the loss exactly as standard packing does."""

import random
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.data.trie_batch import (
    TrieMicroBatches,
    schedule_trie_micro_batches,
    trie_packing_keys,
)
from miles.backends.training_utils.data.trie_packing import (
    TrieModelSpec,
    micro_batch_trie_layout,
    routing_aware_packing_keys,
)
from miles.backends.training_utils.loss.hub import math_utils
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.ray.rollout.train_data_conversion import _step_block_partitions
from miles.utils.arguments import validate_trie_packing_args
from miles.utils.output_store.plan import RoutedExpertsSegments


def _rollout(seed: int, num_prompts: int = 2, group_size: int = 3) -> dict:
    """GRPO-shaped samples: a shared system prompt, one prompt per group, then responses."""
    rng = random.Random(seed)
    system = [rng.randrange(1, 50) for _ in range(17)]
    tokens, total_lengths, response_lengths, loss_masks = [], [], [], []
    for _ in range(num_prompts):
        prompt = system + [rng.randrange(1, 50) for _ in range(rng.randrange(3, 9))]
        for _ in range(group_size):
            # a small alphabet, so responses often start alike and share their first tokens
            response = [rng.randrange(1, 4) for _ in range(rng.randrange(2, 7))]
            tokens.append(torch.tensor(prompt + response))
            total_lengths.append(len(prompt) + len(response))
            response_lengths.append(len(response))
            loss_masks.append(torch.tensor([rng.randrange(2) for _ in response], dtype=torch.int))
    return {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }


@pytest.fixture
def on_cpu(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=1), tp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)


def _trie_iterator(rollout: dict, micro_batch_indices: list[list[int]] | None = None) -> data_utils.DataIterator:
    micro_batch_indices = micro_batch_indices or [list(range(len(rollout["tokens"])))]
    packing_keys = trie_packing_keys(rollout)
    trie_micro_batches = TrieMicroBatches(rollout, TrieModelSpec(), micro_batch_indices, packing_keys=packing_keys)
    return data_utils.DataIterator(
        rollout, micro_batch_indices=micro_batch_indices, trie_micro_batches=trie_micro_batches
    )


def _trie_batch(rollout: dict, iterator: data_utils.DataIterator | None = None) -> dict:
    return data_utils.get_batch(iterator or _trie_iterator(rollout), list(rollout), pad_multiplier=8, qkv_format="thd")


def test_a_micro_batch_becomes_one_trie_row(on_cpu):
    rollout = _rollout(0)
    batch = _trie_batch(rollout)
    layout = batch["trie_layout"]

    assert batch["tokens"].shape == (1, layout.num_tokens) and layout.num_tokens % 8 == 0
    assert layout.num_tokens < sum(rollout["total_lengths"])
    assert batch["cu_seqlens_host"] == (0, layout.num_tokens)
    assert torch.equal(batch["position_ids"][0], layout.positions)


def _sample_token_indices(layout) -> list[torch.Tensor]:
    """Each sample's token indices, position by position: the leading keys of its last token's segment."""
    bounds = layout.kv.cu_seqlens.tolist()
    token_indices = {
        sample: layout.kv.index[bounds[segment] : bounds[segment] + length]
        for sample, segment, length in zip(
            layout.sample_indices, layout.sample_segments, layout.sample_lengths, strict=True
        )
    }
    return [token_indices[sample] for sample in range(len(token_indices))]


def test_the_kept_logit_rows_predict_each_response(on_cpu):
    """Row ``p`` of a sample predicts its token ``p + 1``: the kept rows must be each response's, back to back,
    a row several samples share appearing once for each."""
    rollout = _rollout(1)
    batch = _trie_batch(rollout)
    layout, rows = batch["trie_layout"], batch["trie_logit_rows"]
    lengths = rollout["response_lengths"]
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    assert batch["response_row_starts"] == starts

    trained = torch.zeros(layout.num_tokens, dtype=torch.int)
    for token_indices, tokens, total, length, loss_mask, start in zip(
        _sample_token_indices(layout),
        rollout["tokens"],
        rollout["total_lengths"],
        lengths,
        rollout["loss_masks"],
        starts,
        strict=True,
    ):
        assert torch.equal(layout.tokens[token_indices], tokens)
        assert torch.equal(layout.positions[token_indices], torch.arange(total))
        assert torch.equal(rows[start : start + length], token_indices[total - length - 1 : total - 1])
        trained[token_indices[total - length :]] |= loss_mask
    assert len(set(rows.tolist())) < len(rows)
    assert torch.equal(batch["input_loss_masks"][0], trained)


def test_every_pass_reuses_each_micro_batchs_trie_row(on_cpu):
    rollout = _rollout(2)
    num_samples = len(rollout["tokens"])
    halves = [list(range(num_samples // 2)), list(range(num_samples // 2, num_samples))]
    iterator = _trie_iterator(rollout, halves)
    first = [_trie_batch(rollout, iterator)["trie_layout"] for _ in halves]
    iterator.reset()

    for half, layout in zip(halves, first, strict=True):
        assert _trie_batch(rollout, iterator)["trie_layout"] is layout
        assert layout.num_tokens >= max(rollout["total_lengths"][i] for i in half)
        assert len(layout.sample_indices) == len(half)


def test_a_sample_without_a_prompt_token_is_refused():
    with pytest.raises(ValueError, match="at least one prompt token"):
        micro_batch_trie_layout([[5, 6, 7]], [3], [3], TrieModelSpec(), window_start_multiple=1, pad_multiple=1)


def _naive_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None, debug_unified_grad_fused_logprob=False):
    return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))


def test_response_only_logits_score_each_response_as_packed_logits_do(monkeypatch):
    monkeypatch.setattr(math_utils, "compute_log_probs", _naive_compute_log_probs)
    make_parallel_state()
    args = make_args(true_on_policy_mode=False, rollout_temperature=1.0, log_probs_chunk_size=4)
    rollout = _rollout(3)
    totals, responses = rollout["total_lengths"], rollout["response_lengths"]
    packed = torch.randn(1, sum(totals), 64, generator=torch.Generator().manual_seed(0))
    offsets = [sum(totals[:i]) for i in range(len(totals))]
    rows = torch.cat(
        [torch.arange(o + t - r - 1, o + t - 1) for o, t, r in zip(offsets, totals, responses, strict=True)]
    )
    kwargs = dict(args=args, unconcat_tokens=rollout["tokens"], total_lengths=totals, response_lengths=responses)

    expected = get_log_probs_and_entropy(packed, **kwargs)
    starts = [sum(responses[:i]) for i in range(len(responses))]
    actual = get_log_probs_and_entropy(packed[:, rows], response_row_starts=starts, **kwargs)

    for got, want in zip(actual["log_probs"], expected["log_probs"], strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    # Read as packed logits, the response-only rows run out before the last sample.
    with pytest.raises(AssertionError):
        get_log_probs_and_entropy(packed[:, rows], **kwargs)


class _PeerMax:
    """``all_reduce(MAX)`` against one DP peer that packed ``peer_counts`` tries per step."""

    def __init__(self, peer_counts):
        self.peer_counts = peer_counts

    def all_reduce(self, tensor, group, op):
        tensor.copy_(torch.maximum(tensor, torch.tensor(self.peer_counts, dtype=tensor.dtype)))


def _schedule(monkeypatch, rollout, *, peer_counts, num_steps=1, max_tokens=10_000, model_spec=None):
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_utils.GeneralPGUtil, "create", lambda group: _PeerMax(peer_counts))
    return schedule_trie_micro_batches(
        rollout,
        model_spec or TrieModelSpec(),
        num_steps=num_steps,
        num_local_gbs=len(rollout["tokens"]) // num_steps,
        max_tokens_per_gpu=max_tokens,
        dp_group=None,
    )


def _row_sizes(trie_micro_batches: TrieMicroBatches) -> list[int]:
    return [
        trie_micro_batches.layout(index, pad_multiple=1).num_tokens
        for index in range(len(trie_micro_batches.micro_batch_indices))
    ]


def test_a_step_packs_into_tries_within_the_token_budget(monkeypatch):
    rollout = _rollout(4, num_prompts=4)
    budget = max(rollout["total_lengths"]) + 10
    trie_micro_batches, counts = _schedule(monkeypatch, rollout, peer_counts=[0], max_tokens=budget)
    micro_batches = trie_micro_batches.micro_batch_indices

    assert counts == [len(micro_batches)] and len(micro_batches) > 1
    assert sorted(i for mb in micro_batches for i in mb) == list(range(len(rollout["tokens"])))
    assert max(_row_sizes(trie_micro_batches)) <= budget


def test_tries_split_to_match_a_dp_peer_with_more_micro_batches(monkeypatch):
    rollout = _rollout(5, num_prompts=2)
    trie_micro_batches, counts = _schedule(monkeypatch, rollout, peer_counts=[4])
    micro_batches = trie_micro_batches.micro_batch_indices

    assert counts == [4] and len(micro_batches) == 4
    assert sorted(i for mb in micro_batches for i in mb) == list(range(len(rollout["tokens"])))


def test_each_step_keeps_its_samples_but_hands_each_rank_a_contiguous_block():
    num_samples, dp_size, global_batch_size = 21, 2, 8
    blocks = _step_block_partitions(num_samples, dp_size=dp_size, global_batch_size=global_batch_size)
    strided = [list(range(rank, num_samples, dp_size)) for rank in range(dp_size)]
    local = global_batch_size // dp_size

    for step in range(num_samples // global_batch_size):
        window = slice(step * local, (step + 1) * local)
        assert sorted(i for rank in blocks for i in rank[window]) == sorted(
            i for rank in strided for i in rank[window]
        )
        for rank in blocks:
            assert rank[window] == list(range(rank[window][0], rank[window][0] + local))
    assert sorted(i for rank in blocks for i in rank) == list(range(num_samples))


def _trie_args(**overrides):
    values = dict(
        train_backend="fsdp",
        qkv_format="thd",
        true_on_policy_mode=False,
        multi_lora=False,
        loss_type="policy_loss",
        multimodal_token_space_processing=False,
        enable_witness=False,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=8192,
        use_rollout_routing_replay=False,
        use_output_store=False,
    )
    return SimpleNamespace(**(values | overrides))


def test_trie_packing_accepts_a_supported_run():
    validate_trie_packing_args(_trie_args())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"train_backend": "megatron"}, "--train-backend"),
        ({"multi_lora": True}, "--multi-lora-n-adapters"),
        ({"enable_witness": True}, "--enable-witness"),
        ({"loss_type": "custom_loss"}, "custom_loss"),
        ({"use_dynamic_batch_size": False}, "--use-dynamic-batch-size"),
    ],
)
def test_trie_packing_refuses_what_it_does_not_implement(override, message):
    with pytest.raises(ValueError, match=message):
        validate_trie_packing_args(_trie_args(**override))


def test_the_token_budget_holds_for_samples_whose_routing_differs(monkeypatch):
    """Samples their rollouts routed apart share no token; the schedule must not count those tokens as shared."""
    rng = random.Random(0)
    shared = torch.tensor([rng.randrange(1, 50) for _ in range(30)])
    tokens = [shared] * 8
    routed_experts = [torch.full((len(shared) - 1, 2, 2), sample) for sample in range(8)]
    rollout = {
        "tokens": tokens,
        "total_lengths": [len(t) for t in tokens],
        "response_lengths": [3] * len(tokens),
        "loss_masks": [torch.ones(3, dtype=torch.int)] * len(tokens),
        "rollout_routed_experts": routed_experts,
    }
    assert trie_packing_keys(rollout) == routing_aware_packing_keys(tokens, routed_experts)
    budget = 3 * len(shared)
    trie_micro_batches, _ = _schedule(monkeypatch, rollout, peer_counts=[0], max_tokens=budget)

    assert max(_row_sizes(trie_micro_batches)) <= budget


def test_routing_keys_need_the_rows_inline():
    """Without routing replay, or with its rows in the output store (read per micro-batch), samples pack on tokens."""
    rollout = _rollout(4, num_prompts=2)
    assert trie_packing_keys(rollout) is None
    segments = [RoutedExpertsSegments() for _ in rollout["tokens"]]
    assert trie_packing_keys(rollout | {"rollout_routed_experts": segments}) is None
