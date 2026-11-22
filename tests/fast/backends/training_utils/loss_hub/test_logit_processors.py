from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.loss.hub import logit_processors


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,responses,padding",
    [([17, 13, 9, 12], [16, 6, 0, 11], 7), ([5, 4], [4, 4], 0), ([7], [0], 0)],
)
def test_packed_response_views_preserve_values_and_gradients(monkeypatch, dtype, lengths, responses, padding):
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    args = Namespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False)
    logits = torch.randn(1, sum(lengths) + padding, 16, dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    tokens = [torch.arange(length) for length in lengths]
    chunks = list(
        logit_processors._iter_response_chunks(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=lengths,
            response_lengths=responses,
            include_response_indices=True,
        )
    )
    expected = []
    end = 0
    for i, (length, response) in enumerate(zip(lengths, responses, strict=True)):
        end += length
        expected.append(reference[0, end - response - 1 : end - 1])
        torch.testing.assert_close(chunks[i][0], expected[-1], rtol=0, atol=0)
        torch.testing.assert_close(chunks[i][1], tokens[i][-response:] if response else tokens[i][:0])
        assert list(chunks[i][2]) == list(range(response))
    sum((i + 1) * chunk[0].square().sum() for i, chunk in enumerate(chunks)).backward()
    sum((i + 1) * chunk.square().sum() for i, chunk in enumerate(expected)).backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=0)


def test_packed_response_backward_does_not_allocate_a_full_gradient_per_sample(monkeypatch):
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    logits = torch.ones(1, 16 * 17 + 7, 256, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profiler:
        chunks = logit_processors._iter_response_chunks(
            logits,
            args=Namespace(qkv_format="thd", true_on_policy_mode=False, allgather_cp=False),
            unconcat_tokens=[torch.arange(17) for _ in range(16)],
            total_lengths=[17] * 16,
            response_lengths=[16] * 16,
            include_response_indices=False,
        )
        sum(chunk.sum() for chunk, _, _ in chunks).backward()
    gradient_allocations = sum(
        event.cpu_memory_usage for event in profiler.events() if event.name == "aten::slice_backward"
    )
    assert gradient_allocations <= 2 * logits.numel() * logits.element_size()


class TestGetLogProbsAndEntropy:
    def test_stored_log_probs_forward_the_debug_grad_switch_from_training_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fused_cross_entropy_calls: list[tuple[bool, bool]],
    ) -> None:
        """Stored log probabilities forward the debug grad contract from training arguments."""
        parallel_state = SimpleNamespace(
            tp=SimpleNamespace(rank=0, group=None),
            cp=SimpleNamespace(rank=0, size=1),
        )
        monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
        args = Namespace(
            log_probs_backend="torch",
            loss_type="policy_loss",
            qkv_format="thd",
            rollout_temperature=1.0,
            true_on_policy_mode=False,
            log_probs_chunk_size=-1,
            allgather_cp=False,
            debug_unified_grad_fused_logprob=True,
            train_backend="fsdp",
        )

        with torch.no_grad():
            result = logit_processors.get_log_probs_and_entropy(
                torch.randn(1, 4, 8),
                args=args,
                unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
                total_lengths=[4],
                response_lengths=[3],
            )

        assert fused_cross_entropy_calls == [(True, True)]
        assert not result["log_probs"][0].requires_grad
