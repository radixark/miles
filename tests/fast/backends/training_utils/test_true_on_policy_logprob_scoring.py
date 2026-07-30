from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss_hub import logit_processors, math_utils


def _args(*, logprob_dtype: str, bf16: bool, fp16: bool):
    return SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=0.5,
        true_on_policy_mode=True,
        true_on_policy_logprob_dtype=logprob_dtype,
        bf16=bf16,
        fp16=fp16,
    )


def _response_logits(logits: torch.Tensor, args, monkeypatch) -> torch.Tensor:
    monkeypatch.setattr(
        logit_processors,
        "get_parallel_state",
        lambda: SimpleNamespace(cp=SimpleNamespace(size=1)),
    )
    [(response_logits, response_tokens)] = list(
        logit_processors.get_responses(
            logits,
            args=args,
            unconcat_tokens=[torch.tensor([0, 1, 2], dtype=torch.long)],
            total_lengths=[3],
            response_lengths=[2],
        )
    )
    torch.testing.assert_close(response_tokens, torch.tensor([1, 2], dtype=torch.long))
    return response_logits


@pytest.mark.parametrize(
    ("bf16", "fp16", "expected_dtype"),
    [
        (True, False, torch.bfloat16),
        (False, True, torch.float16),
    ],
)
def test_legacy_training_scoring_casts_before_softmax(
    monkeypatch,
    bf16: bool,
    fp16: bool,
    expected_dtype: torch.dtype,
):
    logits = torch.randn(1, 3, 8, dtype=torch.float32)

    response_logits = _response_logits(
        logits,
        _args(logprob_dtype="training", bf16=bf16, fp16=fp16),
        monkeypatch,
    )

    assert response_logits.dtype == expected_dtype


@pytest.mark.parametrize(
    ("bf16", "fp16"),
    [
        (True, False),
        (False, True),
    ],
)
def test_fp32_scoring_upcasts_before_temperature(monkeypatch, bf16: bool, fp16: bool):
    logits = torch.randn(1, 3, 8, dtype=torch.bfloat16)

    response_logits = _response_logits(
        logits,
        _args(logprob_dtype="fp32", bf16=bf16, fp16=fp16),
        monkeypatch,
    )

    expected = logits.float().div(0.5).squeeze(0)[:2]
    assert response_logits.dtype == torch.float32
    assert torch.equal(response_logits, expected)


def test_fp32_logprob_option_does_not_change_value_head_precision(monkeypatch):
    logits = torch.randn(1, 3, 1, dtype=torch.float32)

    response_logits = _response_logits(
        logits,
        _args(logprob_dtype="fp32", bf16=True, fp16=False),
        monkeypatch,
    )

    assert response_logits.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("logprob_dtype", "scoring_dtype"),
    [
        ("training", torch.bfloat16),
        ("fp32", torch.float32),
    ],
)
def test_public_batch_invariant_scorer_uses_requested_dtype(
    monkeypatch,
    logprob_dtype,
    scoring_dtype,
):
    seen = {}

    def fake_batch_invariant_log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        seen["shape"] = input.shape
        seen["dtype"] = input.dtype
        return torch.log_softmax(input, dim=dim)

    parallel_state = SimpleNamespace(
        cp=SimpleNamespace(size=1),
        tp=SimpleNamespace(group=None),
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        math_utils,
        "_load_batch_invariant_log_softmax",
        lambda: fake_batch_invariant_log_softmax,
    )
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=True,
        true_on_policy_logprob_dtype=logprob_dtype,
        true_on_policy_logsoftmax_backend="sglang_batch_invariant",
        bf16=True,
        fp16=False,
        allgather_cp=False,
        log_probs_chunk_size=-1,
        vocab_size=4,
    )
    logits = torch.randn(1, 3, 6, dtype=torch.float32)

    result = logit_processors.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([0, 1, 2], dtype=torch.long)],
        total_lengths=[3],
        response_lengths=[2],
        with_entropy=True,
    )

    assert seen == {"shape": torch.Size([2, 4]), "dtype": scoring_dtype}
    assert result["log_probs"][0].dtype == torch.bfloat16
    assert result["entropy"][0].dtype == scoring_dtype


def test_public_value_head_output_is_unchanged_by_fp32_logprob_option(monkeypatch):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    logits = torch.randn(1, 3, 1, dtype=torch.float32)
    common_args = dict(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=True,
        bf16=True,
        fp16=False,
        allgather_cp=False,
    )

    legacy = logit_processors.get_values(
        logits,
        args=SimpleNamespace(
            **common_args,
            true_on_policy_logprob_dtype="training",
        ),
        unconcat_tokens=[torch.tensor([0, 1, 2], dtype=torch.long)],
        total_lengths=[3],
        response_lengths=[2],
    )
    fp32_logprob_mode = logit_processors.get_values(
        logits,
        args=SimpleNamespace(
            **common_args,
            true_on_policy_logprob_dtype="fp32",
        ),
        unconcat_tokens=[torch.tensor([0, 1, 2], dtype=torch.long)],
        total_lengths=[3],
        response_lengths=[2],
    )

    assert legacy["values"][0].dtype == torch.float32
    assert torch.equal(fp32_logprob_mode["values"][0], legacy["values"][0])
