"""The replay columns of a trainer shard whose arrays are in the output store."""

from argparse import Namespace
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from miles.ray.rollout.train_data_conversion import compact_score_centering_candidates
from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.utils.output_store.plan import StorePlan, StorePlanError, store_reads
from miles.utils.rollout_topk_logprobs import requests_support_logprobs
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


def materialize_store_columns(args: Namespace, rollout_data: dict, *, read: Callable | None) -> None:
    """Replace the shard's rollout_store_plan column with the replay columns the inline path carries.

    ``read`` takes ``[(ref, names)]`` and returns one stash per entry, as ``OutputStore.read_many``
    bound to its stash type does. Reads again on every call, so an FT retry works.
    """
    entries = rollout_data.pop("rollout_store_plan", None)
    if entries is None:
        return
    if read is None:
        raise StorePlanError("the rollout data has output store plans, but this trainer has no output store reader")
    plans = [StorePlan.from_dict(entry) for entry in entries]
    k = args.rollout_top_logprobs_num
    support_topk = k if requests_support_logprobs(args) else 0
    objects = _read_objects(plans, read, support_topk=support_topk)
    columns = [
        plan.materialize(
            objects,
            response_tokens=np.asarray(tokens)[len(tokens) - int(response_length) :],
            support_topk=support_topk,
            indexer_streams=args.backend.rollout_indexer_topk_num_streams,
        )
        for plan, tokens, response_length in zip(
            plans, rollout_data["tokens"], rollout_data["response_lengths"], strict=True
        )
    ]
    # convert gave every plan the same fields, so every sample has the same columns.
    for name in columns[0] if columns else ():
        rollout_data[name] = [sample_columns[name] for sample_columns in columns]
    if support_topk and args.ci_test:
        # Before compaction drops the dense candidate ids.
        _validate_candidates(rollout_data, k)
    compact_score_centering_candidates(args, rollout_data)


def _read_objects(plans: Sequence[StorePlan], read: Callable, *, support_topk: int) -> dict[str, dict[str, Any]]:
    """Read every object the shard's plans take rows from, once."""
    requests = store_reads(plans, support_topk=support_topk)
    try:
        stashes = read(list(requests.values()))
    except (KeyError, ValueError) as e:
        # A missing object or array, or one whose size differs from its ref: no retry can fix it.
        raise StorePlanError(f"cannot read this shard's output store objects: {e}") from e
    return {
        key: {
            name: getattr(stash, name) for name in names
        }  # config-access-exempt: names lists the stash arrays the plan reads
        for (key, (_, names)), stash in zip(requests.items(), stashes, strict=True)
    }


def _validate_candidates(rollout_data: dict, k: int) -> None:
    """The --ci-test check convert runs on inline candidates."""
    for i, tokens in enumerate(rollout_data["tokens"]):
        sample = Sample(
            tokens=list(tokens),
            response_length=int(rollout_data["response_lengths"][i]),
            loss_mask=list(rollout_data["loss_masks"][i]),
            rollout_log_probs=list(rollout_data["rollout_log_probs"][i]),
            rollout_topk_token_ids=rollout_data["rollout_topk_token_ids"][i],
            rollout_topk_log_probs=rollout_data["rollout_topk_log_probs"][i],
            rollout_sampling_mask=RolloutSamplingMask(
                ids=rollout_data["rollout_sampling_mask_ids"][i],
                offsets=rollout_data["rollout_sampling_mask_offsets"][i],
            ),
        )
        validate_rollout_topk_logprobs_sample(sample, k)
