# doc-dev: docs/developer/session-server-sample-assembly.md
"""Training-sample assembly: session records -> `Sample` objects.

Owned by the session package so the assembly can run inside the owning
session worker (records never have to leave the session server); the rollout
driver imports the same functions until the client-side switch lands, so
there is exactly one implementation either way.

- Depends on `generate_utils.generate_endpoint_utils` for the R3 replay
  decoders (accepted utils-level dependency: the decoders have other
  consumers on the single-turn `/generate` path and must not fork).
- Order contract: `truncate_samples_by_total_tokens` runs BEFORE
  `merge_samples` — truncation is a turn-level budget decision (which turns
  survive; the overflowing turn is cut at a turn boundary, later turns are
  dropped) and the turn structure only exists pre-merge.
"""

from argparse import Namespace
from copy import deepcopy

from miles.rollout.generate_utils.generate_endpoint_utils import (
    get_indexer_topk_from_response,
    get_routed_experts_from_response,
)
from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample


def compute_samples_from_openai_records(
    args: Namespace,
    input_sample: Sample,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    max_trim_tokens: int = 0,
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

        sample = _compute_sample_from_openai_record(args, input_sample, record, tokenizer, trim_count)
        samples.append(sample)

    if accumulated_token_ids is not None:
        # Step 5: verify the entire accumulated sequence was consumed
        assert cursor == len(accumulated_token_ids), (
            f"cursor {cursor} != len(accumulated_token_ids) {len(accumulated_token_ids)} "
            f"after processing all {len(records)} records"
        )

    return samples


def _compute_sample_from_openai_record(
    args: Namespace, input_sample: Sample, record: SessionRecord, tokenizer, trim_count: int = 0
) -> Sample:
    choice = record.response["choices"][0]

    prompt_token_ids = record.request.get("input_ids")
    if prompt_token_ids is None:
        raise ValueError("input_ids not found in request — the session server should populate it")

    output_token_ids = [item[1] for item in choice["meta_info"]["output_token_logprobs"]]
    output_log_probs = [item[0] for item in choice["meta_info"]["output_token_logprobs"]]

    sample = deepcopy(input_sample)
    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    sample.rollout_routed_experts = get_routed_experts_from_response(args, choice, sample)
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

    sample.prefix_cache_info.add(choice.get("meta_info", {}))
    if "weight_version" in choice["meta_info"]:
        sample.weight_versions.append(choice["meta_info"]["weight_version"])

    return sample


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
