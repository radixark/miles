"""Packed response selection preserves rows and gradients with one dense backward buffer per microbatch."""

import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.data.context_parallel as cp_utils
import miles.backends.training_utils.loss.hub.logit_processors as logit_processors
from miles.backends.training_utils.data.context_parallel import get_logits_and_tokens_offset_with_cp


def _reference_rows(total_lengths, response_lengths, cp_rank, cp_size, qkv_format, max_seq_lens):
    """Local logit rows per sample, computed with the original nested slicing."""
    rows, end = [], 0
    for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
        if cp_size == 1:
            sample_end = (max_seq_lens[i] * i if max_seq_lens else end) + total_length
            rows.append(list(range(sample_end - response_length - 1, sample_end - 1)))
            end += total_length
            continue
        chunk_size, chunks, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
            total_length,
            response_length,
            qkv_format,
            max_seq_len=max_seq_lens[i] if max_seq_lens is not None else None,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        local = torch.arange(end, end + 2 * chunk_size)
        first = local[:chunk_size][logits_offset[0][0] - chunks[0][0] : logits_offset[0][1] - chunks[0][0]]
        second = local[chunk_size:][logits_offset[1][0] - chunks[1][0] : logits_offset[1][1] - chunks[1][0]]
        rows.append(torch.cat([first, second]).tolist())
        end += 2 * chunk_size
    return rows


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "total_lengths,response_lengths",
    [([9, 16], [4, 1]), ([12], [0]), ([16], [15]), ([7, 5, 20], [6, 2, 13])],
)
def test_shared_span_views_match_original_slices(
    monkeypatch, cp_size, qkv_format, dtype, total_lengths, response_lengths
):
    for cp_rank in range(cp_size):
        _check_response_selection(monkeypatch, cp_rank, cp_size, qkv_format, dtype, total_lengths, response_lengths)


def _check_response_selection(monkeypatch, cp_rank, cp_size, qkv_format, dtype, total_lengths, response_lengths):
    state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda state=state: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda state=state: state)

    padded_length = max((total + 2 * cp_size - 1) // (2 * cp_size) * (2 * cp_size) for total in total_lengths)
    max_seq_lens = [padded_length] * len(total_lengths) if qkv_format == "bshd" else None
    local_rows = (
        len(total_lengths) * padded_length // cp_size
        if max_seq_lens is not None
        else (
            sum(total_lengths)
            if cp_size == 1
            else sum(2 * ((total + 2 * cp_size - 1) // (2 * cp_size)) for total in total_lengths)
        )
    )
    logits = torch.randn(1, local_rows, 5, dtype=dtype, requires_grad=True)
    tokens = [torch.arange(total) for total in total_lengths]
    args = Namespace(qkv_format=qkv_format, true_on_policy_mode=False, allgather_cp=False)

    chunks = list(
        logit_processors._iter_response_chunks(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            include_response_indices=True,
        )
    )

    flat = logits.detach().squeeze(0)
    for (logits_chunk, tokens_chunk, response_indices), rows in zip(
        chunks,
        _reference_rows(total_lengths, response_lengths, cp_rank, cp_size, qkv_format, max_seq_lens),
        strict=True,
    ):
        assert torch.equal(logits_chunk.detach(), flat[rows])
        assert logits_chunk.size(0) == tokens_chunk.size(0) == len(response_indices)
    for (_, tokens_chunk, response_indices), tokens_i, total_length, response_length in zip(
        chunks, tokens, total_lengths, response_lengths, strict=True
    ):
        assert torch.equal(
            tokens_chunk, tokens_i[[total_length - response_length + index for index in response_indices]]
        )

    # One response backward node must serve every packed sample, including CP1.
    assert sum(type(node).__name__ == "_ResponseSpanViewsBackward" for node in _graph_nodes(chunks)) == 1
    sum(chunk.sum() for chunk, _, _ in chunks).backward()
    expected_grad = torch.zeros_like(flat)
    for rows in _reference_rows(total_lengths, response_lengths, cp_rank, cp_size, qkv_format, max_seq_lens):
        expected_grad[rows] += 1
    assert torch.equal(logits.grad.squeeze(0), expected_grad)


def test_inconsistent_cp_padding_fails_before_response_selection(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda state=state: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda state=state: state)
    args = Namespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False)

    with pytest.raises(AssertionError, match="sample 0: local logits have 7 rows.*requires at least 8"):
        list(
            logit_processors._iter_response_chunks(
                torch.randn(1, 7, 5),
                args=args,
                unconcat_tokens=[torch.arange(16)],
                total_lengths=[16],
                response_lengths=[15],
                include_response_indices=True,
            )
        )


def _graph_nodes(chunks):
    pending = [chunk.grad_fn for chunk, _, _ in chunks]
    seen = set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        pending.extend(parent for parent, _ in node.next_functions)
    return seen


@pytest.mark.parametrize("requires_grad", [False, True])
def test_no_grad_response_selection_stays_lazy(monkeypatch, requires_grad):
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda state=state: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda state=state: state)
    visited = []
    original = logit_processors._iter_response_spans

    def record_spans(**kwargs):
        for metadata in original(**kwargs):
            visited.append(metadata)
            yield metadata

    monkeypatch.setattr(logit_processors, "_iter_response_spans", record_spans)
    with torch.no_grad():
        chunks = logit_processors._iter_response_chunks(
            torch.randn(1, 16, 5, requires_grad=requires_grad),
            args=Namespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False),
            unconcat_tokens=[torch.arange(16), torch.arange(16)],
            total_lengths=[16, 16],
            response_lengths=[12, 12],
            include_response_indices=True,
        )
        first, _, _ = next(chunks)
        assert not first.requires_grad
        assert len(visited) == 1
        next(chunks)
        assert len(visited) == 2


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("cp_size", [2, 4, 16])
def test_allgather_cp_packed_response_gradients(monkeypatch, qkv_format, cp_size):
    total_lengths, response_lengths = [19, 33, 12], [9, 0, 11]
    padded = 64
    for cp_rank in range(cp_size):
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size))
        monkeypatch.setattr(logit_processors, "get_parallel_state", lambda state=state: state)
        max_seq_lens = [padded] * 3 if qkv_format == "bshd" else None
        local_length = padded // cp_size
        local_rows = local_length * 3 if max_seq_lens else local_length
        logits = torch.randn(1, local_rows, 5, requires_grad=True)
        chunks = list(
            logit_processors._iter_response_chunks(
                logits,
                args=Namespace(qkv_format=qkv_format, true_on_policy_mode=False, allgather_cp=True),
                unconcat_tokens=[torch.arange(n) for n in total_lengths],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                max_seq_lens=max_seq_lens,
                include_response_indices=True,
            )
        )
        expected = torch.zeros_like(logits)
        sample_start = 0
        for i, ((chunk, tokens, indices), total, response) in enumerate(
            zip(chunks, total_lengths, response_lengths, strict=True)
        ):
            global_start = 0 if max_seq_lens else sample_start
            first = global_start + total - response - 1
            last = global_start + total - 1
            rank_start, rank_end = cp_rank * local_length, (cp_rank + 1) * local_length
            rows = [
                g - rank_start + (i * local_length if max_seq_lens else 0)
                for g in range(first, last)
                if rank_start <= g < rank_end
            ]
            assert torch.equal(chunk.detach(), logits.detach().squeeze(0)[rows])
            assert tokens.tolist() == [total - response + index for index in indices]
            expected[0, rows] = 1
            sample_start += total
        assert sum(type(node).__name__ == "_ResponseSpanViewsBackward" for node in _graph_nodes(chunks)) == 1
        sum(chunk.sum() for chunk, _, _ in chunks).backward()
        assert torch.equal(logits.grad, expected)


def test_empty_microbatch(monkeypatch):
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    assert (
        list(
            logit_processors._iter_response_chunks(
                torch.empty(1, 0, 5, requires_grad=True),
                args=Namespace(qkv_format="thd", true_on_policy_mode=False),
                unconcat_tokens=[],
                total_lengths=[],
                response_lengths=[],
                include_response_indices=True,
            )
        )
        == []
    )


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_packed_sampling_mask_backward(monkeypatch, temperature):
    from miles.utils.sampling_mask import RolloutSamplingMask

    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=1), tp=SimpleNamespace(rank=0, size=1, group=None))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: state)

    def cpu_cross_entropy(logits, tokens, process_group):
        return -logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    monkeypatch.setitem(
        sys.modules,
        "megatron.core.fusions.fused_cross_entropy",
        SimpleNamespace(fused_vocab_parallel_cross_entropy=cpu_cross_entropy),
    )
    leaf = torch.randn(1, 8, 5, requires_grad=True)
    tokens = [torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4])]
    masks = [
        RolloutSamplingMask.from_mask_list([[2, 3], [3, 4]]),
        RolloutSamplingMask.from_mask_list([[3, 4], [0, 4]]),
    ]
    results = logit_processors.get_log_probs_and_entropy(
        leaf + 0,
        args=Namespace(
            qkv_format="thd",
            allgather_cp=False,
            true_on_policy_mode=False,
            rollout_temperature=temperature,
            vocab_size=5,
            log_probs_chunk_size=-1,
            debug_unified_grad_fused_logprob=False,
        ),
        unconcat_tokens=tokens,
        total_lengths=[4, 4],
        response_lengths=[2, 2],
        rollout_sampling_mask=masks,
    )
    actual_loss = sum(lp.sum() for lp in results["log_probs"])
    actual_loss.backward()
    reference = leaf.detach().clone().requires_grad_()
    selected = reference[0, [1, 2, 5, 6]] / temperature
    allowed = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, False, True, True],
            [False, False, False, True, True],
            [True, False, False, False, True],
        ]
    )
    expected = (
        selected.masked_fill(~allowed, float("-inf"))
        .log_softmax(-1)
        .gather(-1, torch.tensor([2, 3, 3, 4]).unsqueeze(-1))
        .squeeze(-1)
    )
    expected.sum().backward()
    torch.testing.assert_close(torch.cat(results["log_probs"]), expected)
    torch.testing.assert_close(leaf.grad, reference.grad)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_allgather_cp_padding_bounds_fail_on_host(monkeypatch, grad_enabled):
    state = SimpleNamespace(cp=SimpleNamespace(rank=1, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: state)
    with torch.set_grad_enabled(grad_enabled), pytest.raises(
        AssertionError, match="sample 1: response spans.*exceed local logits with 4 rows"
    ):
        list(
            logit_processors._iter_response_chunks(
                torch.randn(1, 4, 5, requires_grad=True),
                args=Namespace(qkv_format="bshd", true_on_policy_mode=False, allgather_cp=True),
                unconcat_tokens=[torch.arange(8), torch.arange(8)],
                total_lengths=[8, 8],
                response_lengths=[6, 6],
                max_seq_lens=[8, 8],
                include_response_indices=True,
            )
        )


@pytest.mark.parametrize("cp_size", [1, 2, 16])
def test_unused_sample_gradients_and_input_storage(monkeypatch, cp_size):
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=cp_size))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    total = 2 * cp_size * 8
    local_rows = (total * 2) if cp_size == 1 else 32
    logits = torch.randn(1, local_rows, 5, requires_grad=True)
    chunks = list(
        logit_processors._iter_response_chunks(
            logits,
            args=Namespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False),
            unconcat_tokens=[torch.arange(total), torch.arange(total)],
            total_lengths=[total, total],
            response_lengths=[total - 1, total - 1],
            include_response_indices=True,
        )
    )
    # Only the second sample contributes. Unused outputs must not get materialized
    # zero gradients or cause unselected rows to inherit another sample's gradient.
    chunks[1][0].square().sum().backward()
    expected = torch.zeros_like(logits)
    rows = _reference_rows([total, total], [total - 1, total - 1], 0, cp_size, "thd", None)[1]
    expected[0, rows] = 2 * logits.detach()[0, rows]
    assert torch.equal(logits.grad, expected)
    if cp_size == 1:
        assert chunks[0][0].untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()


def test_span_views_support_second_derivatives():
    logits = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    spans = [(0, 2), (3, 6)]

    def select(tensor):
        return logit_processors._ResponseSpanViews.apply(tensor, spans)

    assert torch.autograd.gradcheck(select, (logits,))
    assert torch.autograd.gradgradcheck(select, (logits,))


def test_training_response_views_reject_inplace_mutation():
    logits = torch.randn(6, 3, requires_grad=True)
    first, _ = logit_processors._ResponseSpanViews.apply(logits, [(0, 2), (3, 6)])
    with pytest.raises(RuntimeError, match="view.*modified inplace"):
        first.div_(2)
