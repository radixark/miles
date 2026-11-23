"""The true-on-policy CI check of rollout top-k candidates against the trainer's distribution."""

from argparse import Namespace

import numpy as np
import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss.hub.logit_processors import get_rollout_topk_mismatch
from miles.backends.training_utils.parallel import GroupInfo, ParallelState

PROMPT_LEN, RESPONSE_LEN, VOCAB, K = 2, 4, 16, 4
LOSS_MASK = [1, 1, 0, 1]  # row 2 is a multi-turn observation without candidates


@pytest.fixture(autouse=True)
def single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    singleton = GroupInfo(rank=0, size=1, group=None)
    monkeypatch.setattr(
        parallel,
        "_parallel_state",
        ParallelState(
            **{name: singleton for name in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
        ),
    )


def _args(log_probs_chunk_size: int = -1) -> Namespace:
    return Namespace(
        train_backend="fsdp",
        backend=Namespace(bf16=False, fp16=False),
        true_on_policy_mode=True,
        log_probs_chunk_size=log_probs_chunk_size,
        use_sampling_support_replay=False,
        rollout_temperature=1.0,
        qkv_format="thd",
        allgather_cp=False,
    )


def _logits() -> torch.Tensor:
    return torch.randn(1, PROMPT_LEN + RESPONSE_LEN, VOCAB, generator=torch.Generator().manual_seed(0))


def _trainer_log_probs(logits: torch.Tensor) -> torch.Tensor:
    # response row r is scored by the logits at position PROMPT_LEN - 1 + r
    return logits[0, PROMPT_LEN - 1 : PROMPT_LEN - 1 + RESPONSE_LEN].log_softmax(-1)


def _batch(ids: np.ndarray, logps: np.ndarray) -> dict:
    ids, logps = ids.copy(), logps.copy()
    ids[2], logps[2] = -1, -np.inf
    return {
        "unconcat_tokens": [torch.arange(PROMPT_LEN + RESPONSE_LEN)],
        "total_lengths": [PROMPT_LEN + RESPONSE_LEN],
        "response_lengths": [RESPONSE_LEN],
        "loss_masks": [torch.tensor(LOSS_MASK)],
        "rollout_topk_token_ids": [ids],
        "rollout_topk_log_probs": [logps],
    }


def _sampler_candidates(logits: torch.Tensor, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    top = _trainer_log_probs(logits).topk(k, dim=-1)
    return top.indices.numpy().astype(np.int32), top.values.numpy()


def _check(
    logits: torch.Tensor, ids: np.ndarray, logps: np.ndarray, log_probs_chunk_size: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    result = get_rollout_topk_mismatch(logits, args=_args(log_probs_chunk_size), batch=_batch(ids, logps))
    (abs_diff,), (set_mismatch,) = result["topk_logprob_abs_diff"], result["topk_set_mismatch"]
    return abs_diff, set_mismatch


def test_matching_candidates_give_zero() -> None:
    logits = _logits()
    abs_diff, set_mismatch = _check(logits, *_sampler_candidates(logits))
    assert abs_diff.tolist() == [0.0] * RESPONSE_LEN
    assert set_mismatch.tolist() == [0.0] * RESPONSE_LEN


def test_tie_order_does_not_matter() -> None:
    logits = _logits()
    ids, logps = _sampler_candidates(logits)
    abs_diff, set_mismatch = _check(logits, ids[:, ::-1], logps[:, ::-1])
    assert abs_diff.sum() == 0 and set_mismatch.sum() == 0


# 3 splits the 4 response rows unevenly; row 1 lands in the first chunk, row 3 alone in the second
@pytest.mark.parametrize("log_probs_chunk_size", [-1, 3])
def test_perturbed_candidate_log_prob_is_reported(log_probs_chunk_size: int) -> None:
    logits = _logits()
    ids, logps = _sampler_candidates(logits)
    logps[1, 2] += 1e-3
    abs_diff, set_mismatch = _check(logits, ids, logps, log_probs_chunk_size)
    assert abs_diff[1] == pytest.approx(1e-3, rel=1e-3)
    assert set_mismatch.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert abs_diff[[0, 2, 3]].tolist() == [0.0, 0.0, 0.0]


def test_swapped_candidate_set_is_reported() -> None:
    # the k-th candidate is replaced by the (k+1)-th with its exact trainer log-prob: values match per id,
    # but the set is not the trainer's top-k
    logits = _logits()
    ids, logps = _sampler_candidates(logits, K + 1)
    ids = np.concatenate((ids[:, : K - 1], ids[:, K:]), axis=-1)
    logps = np.concatenate((logps[:, : K - 1], logps[:, K:]), axis=-1)
    abs_diff, set_mismatch = _check(logits, ids, logps)
    assert abs_diff.sum() == 0
    assert set_mismatch.tolist() == [1.0, 1.0, 0.0, 1.0]  # the observation row stays out of it


@pytest.mark.parametrize("train_backend", ["fsdp", "torchtitan", "megatron"])
def test_backend_vocabulary_scope_preserves_matching_candidates(train_backend: str) -> None:
    """Only Megatron trims padded vocabulary logits using its backend configuration."""
    args = _args()
    args.train_backend = train_backend
    logits = _logits()
    if train_backend == "megatron":
        args.backend.vocab_size = VOCAB - 3
        logits[..., VOCAB - 3 :] = 100.0
        scoring_logits = logits[..., : VOCAB - 3]
    else:
        scoring_logits = logits
    ids, logps = _sampler_candidates(scoring_logits)

    result = get_rollout_topk_mismatch(logits, args=args, batch=_batch(ids, logps))

    assert result["topk_logprob_abs_diff"][0].tolist() == [0.0] * RESPONSE_LEN
    assert result["topk_set_mismatch"][0].tolist() == [0.0] * RESPONSE_LEN
