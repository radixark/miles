"""Generate a DUMMY ``--dump-details`` directory through miles'
REAL dumping pipeline.

Only the *content* is fake (tiny random rollouts); every file on disk is
produced by the actual production code path, so if a PR changes the dumping
logic — schemas, file naming, DP partition / ``sample_indices`` semantics,
reward post-processing, the batch-global ``raw_reward`` layout — dumps built
here change with it and the reader tests break loudly instead of silently
validating a stale hand-written copy.

Real code invoked per step, in the real call order:

1. ``save_debug_rollout_data``            (rollout + eval dump files)
2. ``convert_samples_to_train_data``      (column dict incl. ``sample_indices``)
3. ``split_train_data_by_dp_local``       (real seqlen-balanced DP partition)
4. ``process_rollout_data_shard``         (train-side partition pop / reorder)
5. ``save_debug_train_data_for_rank``     (per-rank train dump files)

The only hand-synthesized part is :func:`_apply_dummy_actor_columns`: the
training actor's GPU forward passes cannot run in a CPU test, so the *values*
of its per-token outputs (``log_probs``, ``ref_log_probs``, ``entropy``,
``ref_entropy``, ``advantages``, ``returns``) are random tensors and the
actor-side tensorization of ``tokens``/``loss_masks``/``rollout_log_probs``
is mimicked. Column names there mirror real dumps; the end-to-end guard for
them is the realdata test plus the e2e CI run.
"""

from __future__ import annotations

import os
import random
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import torch

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data,
    process_rollout_data_shard,
    split_train_data_by_dp_local,
)
from miles.utils.train_dump_utils import save_debug_train_data_for_rank
from miles.utils.types import Sample


@dataclass
class DummyRunTruth:
    """Ground truth returned by :func:`dump_dummy_run` for test assertions."""

    n_samples_per_step: int
    # step -> one list of Sample.index per DP shard, in the dumped row order
    # (taken from the REAL partition output, not recomputed)
    shard_indices: dict[int, list[list[int]]]
    eval_ids: list[int]


def dump_dummy_run(
    dump_dir: Path,
    *,
    steps: int = 2,
    num_prompts: int = 4,
    n_samples_per_prompt: int = 2,
    dp_size: int = 2,
    tp_dup: int = 2,
    max_response_len: int = 16,
    with_entropy: bool = True,
    with_eval: bool = True,
    with_tokenizer: bool = True,
    remove_sample_indices: tuple[int, ...] = (),
    seed: int = 0,
) -> DummyRunTruth:
    """``remove_sample_indices`` marks the given within-step positions as
    ``remove_sample=True`` in every step; the real conversion then writes an
    all-zero loss mask for them."""
    n = num_prompts * n_samples_per_prompt
    assert n % dp_size == 0, "equal-size DP partition needs divisible batch"
    args = _make_args(dump_dir, num_prompts=num_prompts, n_samples_per_prompt=n_samples_per_prompt)
    rng = random.Random(seed)
    truth = DummyRunTruth(n_samples_per_step=n, shard_indices={}, eval_ids=[])

    if with_tokenizer:
        _write_tokenizer(dump_dir)

    for rollout_id in range(steps):
        samples = [
            _make_sample(
                rng,
                index=rollout_id * n + i,
                group_index=i // n_samples_per_prompt,
                max_response_len=max_response_len,
                remove_sample=i in remove_sample_indices,
            )
            for i in range(n)
        ]
        save_debug_rollout_data(args, samples, rollout_id=rollout_id, evaluation=False)
        if with_eval and rollout_id == 0:
            # eval takes the real input shape: {dataset_name: {"samples": [...]}}
            eval_data = {"dummy_eval": {"samples": samples[: n // 2]}}
            save_debug_rollout_data(args, eval_data, rollout_id=rollout_id, evaluation=True)
            truth.eval_ids.append(rollout_id)

        train_data = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        shards = split_train_data_by_dp_local(args, train_data, dp_size)
        truth.shard_indices[rollout_id] = [list(shard["sample_indices"]) for shard in shards]
        generator = torch.Generator().manual_seed(rng.randint(0, 2**31))
        for shard_index, shard in enumerate(shards):
            shard = process_rollout_data_shard(args, shard)
            _apply_dummy_actor_columns(shard, generator, with_entropy=with_entropy)
            for dup in range(tp_dup):
                save_debug_train_data_for_rank(
                    args, rollout_id=rollout_id, rollout_data=shard, rank=shard_index * tp_dup + dup
                )

    age_files(dump_dir)
    return truth


def _make_args(dump_dir: Path, *, num_prompts: int, n_samples_per_prompt: int) -> Namespace:
    # Exactly the attributes the invoked dump-pipeline functions read; a new
    # attribute dependency in main code fails loudly here (drift detection).
    return Namespace(
        save_debug_rollout_data=f"{dump_dir}/rollout_data/{{rollout_id}}.pt",
        save_debug_train_data=f"{dump_dir}/train_data/{{rollout_id}}_{{rank}}.pt",
        balance_data=True,
        use_dynamic_global_batch_size=False,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        rollout_batch_size=num_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        reward_key=None,
    )


def _make_sample(
    rng: random.Random, *, index: int, group_index: int, max_response_len: int, remove_sample: bool = False
) -> Sample:
    response_length = rng.randint(4, max_response_len)
    prompt_length = rng.randint(3, 8)
    truncated = response_length == max_response_len
    return Sample(
        group_index=group_index,
        index=index,
        prompt="What is 1+1?",
        tokens=[rng.randint(0, _VOCAB_SIZE - 1) for _ in range(prompt_length + response_length)],
        response="x" * response_length,
        response_length=response_length,
        reward=float(rng.random() < 0.5),
        rollout_log_probs=[-rng.random() for _ in range(response_length)],
        weight_versions=[str(index % 3)],
        status=Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED,
        remove_sample=remove_sample,
    )


_VOCAB_SIZE = 100


def _write_tokenizer(dump_dir: Path) -> None:
    """Persist a tiny word-level tokenizer (token id i <-> text "t{i}") with the
    same ``save_pretrained`` mechanism the real data source uses for
    ``{dump_details}/tokenizer/``."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(_VOCAB_SIZE)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=raw).save_pretrained(dump_dir / "tokenizer")


def _apply_dummy_actor_columns(shard: dict, generator: torch.Generator, *, with_entropy: bool) -> None:
    """Mimic what the training actor does to a shard before dumping it:
    tensorize the converted columns and attach forward-pass outputs.
    The values are random; the column names mirror real dumps."""
    response_lengths = shard["response_lengths"]

    def per_token(sign: float = 1.0) -> list[torch.Tensor]:
        return [sign * torch.rand(r, generator=generator) for r in response_lengths]

    shard["tokens"] = [torch.tensor(t) for t in shard["tokens"]]
    shard["loss_masks"] = [torch.tensor(m, dtype=torch.int32) for m in shard["loss_masks"]]
    shard["rollout_log_probs"] = [torch.tensor(p) for p in shard["rollout_log_probs"]]
    shard["log_probs"] = per_token(sign=-1.0)
    shard["ref_log_probs"] = per_token(sign=-1.0)
    shard["advantages"] = per_token()
    shard["returns"] = per_token()
    if with_entropy:
        shard["entropy"] = per_token()
        shard["ref_entropy"] = per_token()


def age_files(dump_dir: Path, *, seconds: float = 100.0) -> None:
    """Push mtimes into the past so files pass the reader's visibility guard."""
    stamp = time.time() - seconds
    for path in dump_dir.rglob("*.pt"):
        os.utime(path, (stamp, stamp))
