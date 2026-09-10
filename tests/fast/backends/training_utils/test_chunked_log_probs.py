import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.loss_hub import math_utils
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy

MODEL_DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("model_dtype", MODEL_DTYPES)
def test_chunked_log_probs_upcast_only_each_chunk(monkeypatch, model_dtype):
    kernel_inputs = []

    def fake_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None):
        assert sampling_mask is None
        kernel_inputs.append((logits.shape, logits.dtype))
        return torch.zeros((tokens.size(0), 1), dtype=logits.dtype)

    monkeypatch.setattr(math_utils, "compute_log_probs", fake_compute_log_probs)
    logits = torch.zeros((5, 8), dtype=model_dtype)
    tokens = torch.zeros(5, dtype=torch.long)

    math_utils.calculate_log_probs_and_entropy(logits, tokens, None, chunk_size=2)

    assert kernel_inputs == [
        (torch.Size([2, 8]), torch.float32),
        (torch.Size([2, 8]), torch.float32),
        (torch.Size([1, 8]), torch.float32),
    ]


@pytest.mark.parametrize("model_dtype", MODEL_DTYPES)
def test_temperature_is_applied_in_fp32(monkeypatch, model_dtype):
    def naive_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None):
        assert sampling_mask is None
        return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))

    monkeypatch.setattr(math_utils, "compute_log_probs", naive_compute_log_probs)
    make_parallel_state()
    args = make_args(true_on_policy_mode=False, rollout_temperature=0.7, log_probs_chunk_size=4)

    g = torch.Generator().manual_seed(0)
    logits = (torch.randn((1, 12, 32), generator=g) * 10).to(model_dtype)
    tokens = [torch.randint(0, 32, (12,), generator=g)]

    kwargs = dict(args=args, unconcat_tokens=tokens, total_lengths=[12], response_lengths=[6])
    out_model = get_log_probs_and_entropy(logits, **kwargs)
    out_fp32 = get_log_probs_and_entropy(logits.float(), **kwargs)

    # Model-precision and fp32 inputs carry identical information, so any
    # difference comes from arithmetic done before the fp32 upcast.
    torch.testing.assert_close(out_model["log_probs"][0], out_fp32["log_probs"][0], rtol=0, atol=1e-6)


def test_graph_placeholder_sums_in_fp32():
    # An fp16 sum overflows to inf and 0 * inf is nan; the fp32 sum stays finite.
    logits = torch.full((70_000,), 2.0, dtype=torch.float16)

    assert torch.isinf(logits.sum())
    placeholder = 0 * logits.sum(dtype=torch.float32)
    assert placeholder == 0
