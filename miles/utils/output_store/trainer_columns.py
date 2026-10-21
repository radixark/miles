"""The replay columns of a trainer shard whose arrays are in the output store."""

from argparse import Namespace
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from miles.ray.rollout.train_data_conversion import _compact_candidate_ids
from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.utils.output_store.plan import StorePlan, StorePlanError
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


def materialize_store_columns(args: Namespace, rollout_data: dict, *, read: Callable | None) -> None:
    """Replace the shard's rollout_store_plan column with the replay columns the inline path carries.

    ``read`` takes ``[(ref, names)]`` and returns one stash per entry, as ``OutputStore.read_many``
    bound to its stash type and buffer size does. Reads again on every call, so an FT retry works.
    """
    entries = rollout_data.pop("rollout_store_plan", None)
    if entries is None:
        return
    if read is None:
        raise StorePlanError("the rollout data has output store plans, but this trainer has no output store reader")
    plans = [StorePlan.from_dict(entry) for entry in entries]
    k = args.rollout_top_logprobs_num
    # Requests asked SGLang for support log-probs exactly in support mode with top-k collected.
    support_topk = k if args.rollout_sampling_logprobs_mode == "support" else 0
    objects = _read_objects(plans, read)
    columns = [
        plan.materialize(
            objects,
            response_tokens=np.asarray(tokens)[len(tokens) - int(response_length) :],
            support_topk=support_topk,
            indexer_streams=getattr(args, "rollout_indexer_topk_num_streams", None),
        )
        for plan, tokens, response_length in zip(
            plans, rollout_data["tokens"], rollout_data["response_lengths"], strict=True
        )
    ]
    # convert gave every plan the same fields, so every sample has the same columns.
    for name in columns[0] if columns else ():
        rollout_data[name] = [sample_columns[name] for sample_columns in columns]
    if support_topk:
        if args.ci_test:
            # Before compaction drops the dense candidate ids.
            _validate_candidates(rollout_data, k)
        if args.loss_type == "score_centering":
            _compact_candidate_ids(rollout_data)


def _read_objects(plans: Sequence[StorePlan], read: Callable) -> dict[str, dict[str, Any]]:
    """Read every object the shard's plans take rows from, once."""
    requests: dict[str, tuple[Any, list[str]]] = {}
    for plan in plans:
        for key, (ref, names) in plan.reads().items():
            merged = requests.setdefault(key, (ref, []))[1]
            merged += [name for name in names if name not in merged]
    try:
        stashes = read(list(requests.values()))
    except (KeyError, ValueError) as e:
        # A missing object or array, or an array larger than the buffer: no retry can fix it.
        raise StorePlanError(f"cannot read this shard's output store objects: {e}") from e
    return {
        key: {name: getattr(stash, name) for name in names}
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
