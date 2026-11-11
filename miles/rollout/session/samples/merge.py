"""Training-sample assembly: session records -> per-turn `Sample`s, truncated at turn boundaries.

Owned by the session package so the assembly runs on the owning instance (records never have to leave the session server). The wire codec for the assembled reply lives in `codec`.

- Depends on `generate_utils.generate_endpoint_utils` for the R3 replay decoders (accepted utils-level dependency: the decoders have other consumers on the single-turn `/generate` path and must not fork).
- Order contract: `truncate_samples_by_total_tokens` runs BEFORE `merge_samples` — truncation is a turn-level budget decision (which turns survive; the overflowing turn is cut at a turn boundary, later turns are dropped) and the turn structure only exists pre-merge.
- Additional R3 patches stay out of per-turn `Sample`s; after the ordinary merge selects the final token prefix, the session assembler decodes and concatenates only the patches needed for that prefix.
"""

from argparse import Namespace
from copy import deepcopy
from dataclasses import replace

import numpy as np

from miles.rollout.generate_utils.generate_endpoint_utils import (
    get_indexer_topk_from_response,
    get_routed_experts_from_response,
)
from miles.rollout.generate_utils.rollout_topk_logprobs import append_rollout_topk_logprobs
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.generate_utils.sampling_mask import append_sampling_metadata, read_sampled_logprobs
from miles.rollout.session.types import SessionRecord
from miles.utils.lifecycle import attach_lifecycle_metadata
from miles.utils.output_store.plan import (
    OUTPUT_STORE_REF_KEY,
    OUTPUT_STORE_REQUEST_KEY,
    RoutedExpertsSegments,
    StorePlan,
    store_ref_from_dict,
)
from miles.utils.types import Sample, WeightVersionsPerCall


def compute_samples_from_openai_records(
    args: Namespace,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    max_trim_tokens: int = 0,
    *,
    use_addition_r3: bool = False,
    evaluation: bool = False,
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
            args, record, tokenizer, trim_count, use_addition_r3=use_addition_r3, evaluation=evaluation
        )
        attach_lifecycle_metadata(sample, record, records[i - 1] if i else None, turn=i + 1)
        if is_last and args.save_debug_trajectory_data is not None:
            # A copy: sample hooks may edit the exported messages, which must not reach the session's records.
            sample.metadata["messages"] = deepcopy(
                record.request["messages"] + [record.response["choices"][0]["message"]]
            )
        samples.append(sample)

    if accumulated_token_ids is not None:
        # Step 5: verify the entire accumulated sequence was consumed
        assert cursor == len(accumulated_token_ids), (
            f"cursor {cursor} != len(accumulated_token_ids) {len(accumulated_token_ids)} "
            f"after processing all {len(records)} records"
        )

    return samples


def _compute_sample_from_openai_record(
    args: Namespace,
    record: SessionRecord,
    tokenizer,
    trim_count: int = 0,
    *,
    use_addition_r3: bool = False,
    evaluation: bool = False,
) -> Sample:
    choice = record.response["choices"][0]
    finish_reason = choice.get("finish_reason")

    prompt_token_ids = record.request.get("input_ids")
    if prompt_token_ids is None:
        raise ValueError("input_ids not found in request — the session server should populate it")

    output_token_ids = [item[1] for item in choice["meta_info"]["output_token_logprobs"]]
    output_log_probs = [item[0] for item in choice["meta_info"]["output_token_logprobs"]]

    sample = Sample()
    via_store = record.request.get(OUTPUT_STORE_REQUEST_KEY, False)
    sampling_logprobs_mode = record.request.get("sampling_logprobs_mode", "selected")
    if record.request.get("return_sampling_mask", False):
        if via_store:
            # Only the mask rows are stored; the sampled tokens' log-probs, one per token, stay in meta_info.
            output_log_probs = read_sampled_logprobs(
                choice["meta_info"],
                output_token_ids,
                sampling_logprobs_mode=sampling_logprobs_mode,
            )
        else:
            output_log_probs = append_sampling_metadata(
                sample,
                output_token_ids,
                choice["meta_info"],
                aborted=finish_reason == "abort",
                sampling_logprobs_mode=sampling_logprobs_mode,
            )
    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    # With the output store, the trainer builds support-mode candidates from the stored mask.
    if not evaluation and (
        record.request.get("top_logprobs") or (sampling_logprobs_mode == "support" and not via_store)
    ):
        append_rollout_topk_logprobs(
            sample,
            choice["meta_info"],
            args.rollout_top_logprobs_num,
            sampling_logprobs_mode=sampling_logprobs_mode,
        )
    sample.rollout_routed_experts = (
        None if use_addition_r3 else get_routed_experts_from_response(args, choice, len(sample.tokens) - 1)
    )
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, choice, sample)
    if via_store:
        # Rows of the untrimmed turn, so the TITO trim below cuts the plan like the tokens. Routed experts
        # come from the records after the merge (adopt_session_records); a copy, since turn_args_history
        # shares the request dict. A record never holds an abort: the session rejects it before recording.
        sample.rollout_store_plan = StorePlan.from_response(
            store_ref_from_dict(choice["meta_info"].get(OUTPUT_STORE_REF_KEY)),
            request={**record.request, RoutedExpertsSegments.request_key: False},
            num_tokens=len(sample.tokens),
            response_length=len(output_token_ids),
        )
    sample.weight_versions = [WeightVersionsPerCall.from_meta_info(choice["meta_info"], output_end=len(sample.tokens))]

    if trim_count > 0:
        sample.strip_last_output_tokens(trim_count, tokenizer)

    # TODO unify with Sample.update_from_meta_info
    match finish_reason:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    if args.sglang_speculative_algorithm:
        sample.spec_info.add(choice.get("meta_info", {}))
    sample.prefix_cache_info.add(choice.get("meta_info", {}))

    return sample


def adopt_session_records(merged: Sample, records: list[SessionRecord]) -> None:
    """Complete a merged sample's output store plan from its session's records.

    The per-turn plans leave routed experts out: a TITO trim of a turn would cut a row that the
    next turn's incremental object starts after. Like ``merge_samples_with_addition_r3``, they are
    rebuilt from the untrimmed records, each continuing from its ``routed_experts_start_len``. The
    sample also owns every record's object, so objects of turns the merge dropped are deleted with it.
    """
    plan = merged.rollout_store_plan
    if plan is None:
        return
    choices = [record.response["choices"][0] for record in records]
    refs = [store_ref_from_dict(choice["meta_info"].get(OUTPUT_STORE_REF_KEY)) for choice in choices]
    if any(record.request.get(RoutedExpertsSegments.request_key) for record in records):
        required = len(merged.tokens) - 1
        rows = RoutedExpertsSegments()
        for record, choice, ref in zip(records, choices, refs, strict=True):
            if rows.num_rows() >= required:
                # The merge stopped early: later records lie past the merged tokens.
                break
            num_output = len(choice["meta_info"]["output_token_logprobs"])
            turn = StorePlan.from_response(
                ref,
                request=record.request,
                num_tokens=len(record.request["input_ids"]) + num_output,
                response_length=num_output,
            )
            rows = rows.append(turn.fields[RoutedExpertsSegments.name])
        plan = replace(plan, fields={**plan.fields, RoutedExpertsSegments.name: rows.prefix(required)})
    merged.rollout_store_plan = plan.adopt_refs(ref for ref in refs if ref is not None)
    merged.validate()


def merge_samples_with_addition_r3(
    args: Namespace,
    samples: list[Sample],
    records: list[SessionRecord],
    tokenizer,
) -> Sample:
    """Merge ordinary fields, then materialize the required append-only R3 prefix."""
    merged = merge_samples(samples, tokenizer)
    if all(record.response["choices"][0]["meta_info"].get("routed_experts") is None for record in records):
        return merged

    required_rows = len(merged.tokens) - 1
    covered_rows = 0
    chunks: list[np.ndarray] = []
    for i, record in enumerate(records):
        if chunks and covered_rows >= required_rows:
            break

        choice = record.response["choices"][0]
        info = choice["meta_info"].get("routed_experts")
        if info is None:
            raise ValueError(f"additional R3: record {i} has no routed_experts payload")

        start = record.request.get("routed_experts_start_len")
        if start is None:
            raise ValueError(f"additional R3: record {i} request carries no routed_experts_start_len")
        if start != covered_rows:
            raise ValueError(f"additional R3: record {i} starts at row {start}; expected {covered_rows}")

        end = len(record.request["input_ids"]) + len(choice["meta_info"]["output_token_logprobs"]) - 1
        if end < start:
            raise ValueError(f"additional R3: record {i} has invalid offsets (start={start}, end={end})")
        delta_rows = end - start
        if bool(info) != bool(delta_rows):
            raise ValueError(f"additional R3: record {i} payload presence does not match {delta_rows} rows")

        patch = get_routed_experts_from_response(args, choice, delta_rows)
        if len(patch) or required_rows == 0:
            chunks.append(patch)
        covered_rows = end

    if covered_rows < required_rows:
        raise ValueError(
            f"additional R3 covers {covered_rows} rows but the merged sample needs "
            f"{required_rows} (len(tokens) - 1)"
        )
    merged.rollout_routed_experts = np.concatenate(chunks)[:required_rows]
    return merged


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
            # Later turns are dropped, so mark a COMPLETED last kept turn as cut off.
            if result and result[-1].status == Sample.Status.COMPLETED:
                result[-1].status = Sample.Status.TRUNCATED
            break

        sample.strip_last_output_tokens(overshoot, tokenizer)
        sample.status = Sample.Status.TRUNCATED
        result.append(sample)
        break

    return result
