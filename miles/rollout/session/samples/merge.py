"""Training-sample assembly: session records -> per-turn `Sample`s, truncated at turn boundaries.

Owned by the session package so the assembly runs on the owning instance (records never have to leave the session server). The wire codec for the assembled reply lives in `codec`.

- Depends on `generate_utils.generate_endpoint_utils` for the R3 replay decoders (accepted utils-level dependency: the decoders have other consumers on the single-turn `/generate` path and must not fork).
- Order contract: `truncate_samples_by_total_tokens` runs BEFORE `merge_samples` — truncation is a turn-level budget decision (which turns survive; the overflowing turn is cut at a turn boundary, later turns are dropped) and the turn structure only exists pre-merge.
- Additional R3 (`use_addition_r3`, in-place weight updates): each record's `routed_experts` is a patch of `end - routed_experts_start_len` rows, not a per-turn full tensor, so per-turn decoding is skipped and `merge_samples_with_addition_r3` folds the patches into one full tensor after the merge decides the terminal record. Only the session path owns this: records persist the request offsets the fold needs.
"""

from argparse import Namespace

import numpy as np
import pybase64

from miles.rollout.generate_utils.generate_endpoint_utils import (
    get_indexer_topk_from_response,
    get_routed_experts_from_response,
)
from miles.rollout.generate_utils.sample_utils import merge_samples_with_terminal_index
from miles.rollout.session.types import SessionRecord
from miles.utils.lifecycle import attach_lifecycle_metadata
from miles.utils.types import Sample


def compute_samples_from_openai_records(
    args: Namespace,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    max_trim_tokens: int = 0,
    *,
    use_addition_r3: bool = False,
) -> list[Sample]:
    """Convert per-turn session records into training Samples, aligning each
    turn's output tokens against the TITO accumulated token sequence.

    Each record carries its own ``prompt_token_ids`` and ``output_token_ids``
    (with logprobs).  We want to reuse those per-turn logprobs directly
    instead of re-decoding, but we must first trim "trailing tokens" — stop
    tokens the model emitted that the chat template also renders as the next
    turn's delimiter — to avoid double-counting.

    See ``TestTITOTrailingTokenTrim`` in
    ``tests/fast/rollout/session/test_samples.py``
    for a concrete worked example with token-level walkthroughs.
    """
    samples = []
    cursor = 0

    for i, record in enumerate(records):
        is_last = i == len(records) - 1
        prompt_ids = record.request["input_ids"]
        output_ids = [t[1] for t in record.response["choices"][0]["meta_info"]["output_token_logprobs"]]

        trim_count = 0
        if accumulated_token_ids is not None:
            # Step 1: position cursor right after this turn's prompt
            cursor = len(prompt_ids)

            # Step 2: greedily match output_ids against accumulated[cursor:]
            matched = 0
            for j in range(len(output_ids)):
                idx = cursor + j
                if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
                    matched += 1
                else:
                    break

            # Step 3: unmatched trailing tokens were consumed by the next
            # turn's template rendering (e.g. stop tokens that double as
            # the next message delimiter) — strip them from the sample.
            trim_count = len(output_ids) - matched
            allowed = 0 if is_last else max_trim_tokens
            assert trim_count <= allowed, (
                f"trim_count {trim_count} exceeds allowed={allowed} "
                f"(is_last={is_last}, max_trim_tokens={max_trim_tokens}); "
                f"output_ids[-3:]={output_ids[-3:]}, "
                f"accumulated[cursor:cursor+3]={accumulated_token_ids[cursor:cursor+3]}"
            )

            # Step 4: advance cursor past matched output to the next turn
            cursor += matched

        sample = _compute_sample_from_openai_record(
            args, record, tokenizer, trim_count, use_addition_r3=use_addition_r3
        )
        attach_lifecycle_metadata(sample, record, records[i - 1] if i else None, turn=i + 1)
        if is_last and args.save_debug_trajectory_data is not None:
            sample.metadata["messages"] = record.request["messages"] + [record.response["choices"][0]["message"]]
        samples.append(sample)

    if accumulated_token_ids is not None:
        # Step 5: verify the entire accumulated sequence was consumed
        assert cursor == len(accumulated_token_ids), (
            f"cursor {cursor} != len(accumulated_token_ids) {len(accumulated_token_ids)} "
            f"after processing all {len(records)} records"
        )

    return samples


def _compute_sample_from_openai_record(
    args: Namespace, record: SessionRecord, tokenizer, trim_count: int = 0, *, use_addition_r3: bool = False
) -> Sample:
    choice = record.response["choices"][0]

    prompt_token_ids = record.request.get("input_ids")
    if prompt_token_ids is None:
        raise ValueError("input_ids not found in request — the session server should populate it")

    output_token_ids = [item[1] for item in choice["meta_info"]["output_token_logprobs"]]
    output_log_probs = [item[0] for item in choice["meta_info"]["output_token_logprobs"]]

    sample = Sample()
    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    # An addition-mode response carries an R3 patch, not a per-turn full tensor;
    # merge_samples_with_addition_r3 owns its decoding after the merge.
    sample.rollout_routed_experts = None if use_addition_r3 else get_routed_experts_from_response(args, choice, sample)
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, choice, sample)

    if trim_count > 0:
        sample.strip_last_output_tokens(trim_count, tokenizer)

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    if args.sglang_speculative_algorithm:
        sample.spec_info.add(choice.get("meta_info", {}))
    sample.prefix_cache_info.add(choice.get("meta_info", {}))
    if "weight_version" in choice["meta_info"]:
        sample.weight_versions.append(choice["meta_info"]["weight_version"])

    return sample


def merge_samples_with_addition_r3(
    args: Namespace,
    samples: list[Sample],
    records: list[SessionRecord],
    tokenizer,
) -> Sample:
    """Merge per-turn samples whose records carry additional R3 patches into
    one Sample with a full ``rollout_routed_experts`` tensor.

    ``samples[i]`` must be assembled from ``records[i]`` (compute and turn-level
    truncation preserve that mapping). The merge stop rules are shared with
    ``merge_samples``; the extra record-level rule restores the routed-experts
    replay-gap check that addition-mode samples (R3 field ``None``) can no
    longer express, so a turn without an R3 payload is not consumed.
    """
    present = [_record_routed_experts(record) is not None for record in records]
    merged, terminal = merge_samples_with_terminal_index(
        samples, tokenizer, stop_before=lambda last_consumed, i: present[last_consumed] and not present[i]
    )
    merged.rollout_routed_experts = _fold_addition_routed_experts(
        args, records[: terminal + 1], num_rows=len(merged.tokens) - 1
    )
    return merged


def _record_routed_experts(record: SessionRecord) -> str | None:
    return record.response["choices"][0]["meta_info"].get("routed_experts")


def _fold_addition_routed_experts(
    args: Namespace, records: list[SessionRecord], *, num_rows: int
) -> np.ndarray | None:
    """Decode per-record additional R3 patches and fold them —
    ``R_i = R_(i-1)[:start_i] + delta_i`` — into one ``(num_rows, num_layers,
    topk)`` int32 tensor. Replacement from ``start_i`` supports an offset moving
    backward after rollback or a rewritten suffix. ``num_rows`` is the merged
    sample's ``len(tokens) - 1``; the folded stream must cover at least that
    many rows. Raises ``ValueError`` on a missing payload or offset, wrong
    value count, inconsistent top-k, or a row gap.
    """
    if all(_record_routed_experts(record) is None for record in records):
        return None

    num_layers = args.num_layers
    topk = None
    chunks: list[np.ndarray] = []  # contiguous folded patches, sum of lengths == rows
    rows = 0
    for i, record in enumerate(records):
        info = _record_routed_experts(record)
        if info is None:
            raise ValueError(f"additional R3: record {i} has no routed_experts payload")
        start = record.request.get("routed_experts_start_len")
        if start is None:
            raise ValueError(f"additional R3: record {i} request carries no routed_experts_start_len")
        meta_info = record.response["choices"][0]["meta_info"]
        end = len(record.request["input_ids"]) + len(meta_info["output_token_logprobs"]) - 1
        delta_rows = end - start
        if start < 0 or delta_rows < 0:
            raise ValueError(f"additional R3: record {i} has invalid offsets (start={start}, end={end})")
        if start > rows:
            raise ValueError(f"additional R3: record {i} starts at row {start} but only {rows} rows are retained")
        values = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
        if delta_rows == 0:
            if values.size:
                raise ValueError(f"additional R3: record {i} carries {values.size} values for 0 new rows")
        else:
            if topk is None:
                if values.size == 0 or values.size % (delta_rows * num_layers):
                    raise ValueError(
                        f"additional R3: record {i} has {values.size} values, not a positive multiple of "
                        f"delta_rows * num_layers ({delta_rows} * {num_layers})"
                    )
                topk = values.size // (delta_rows * num_layers)
            elif values.size != delta_rows * num_layers * topk:
                raise ValueError(
                    f"additional R3: record {i} has {values.size} values, expected "
                    f"{delta_rows * num_layers * topk} (delta_rows={delta_rows}, "
                    f"num_layers={num_layers}, topk={topk})"
                )
        # Replacement from `start`: drop retained rows the new patch rewrites.
        while rows > start:
            drop = min(len(chunks[-1]), rows - start)
            chunks[-1] = chunks[-1][: len(chunks[-1]) - drop]
            if not len(chunks[-1]):
                chunks.pop()
            rows -= drop
        if delta_rows:
            chunks.append(values.reshape(delta_rows, num_layers, topk))
        rows = end

    if topk is None:
        # Only empty patches: preserve the existing empty-buffer decode shape.
        topk = 0
    if num_rows > rows:
        raise ValueError(f"additional R3 covers {rows} rows but the merged sample needs {num_rows} (len(tokens) - 1)")
    out = np.empty((num_rows, num_layers, topk), dtype=np.int32)
    pos = 0
    for chunk in chunks:
        if pos >= num_rows:
            break
        take = min(len(chunk), num_rows - pos)
        out[pos : pos + take] = chunk[:take]
        pos += take
    assert pos == num_rows, f"additional R3 fold left rows [{pos}, {num_rows}) uncovered"
    return out


def truncate_samples_by_total_tokens(
    samples: list[Sample],
    max_seq_len: int,
    tokenizer,
) -> list[Sample]:
    """Truncate samples so the total token count (prompt + output, including
    env responses) does not exceed ``max_seq_len``.
    """
    result: list[Sample] = []

    for sample in samples:
        total = len(sample.tokens)
        if total <= max_seq_len:
            result.append(sample)
            continue

        overshoot = total - max_seq_len
        allowed_output = sample.response_length - overshoot
        if allowed_output <= 0:
            break

        sample.strip_last_output_tokens(overshoot, tokenizer)
        sample.status = Sample.Status.TRUNCATED
        result.append(sample)
        break

    return result
