"""Custom generate function: agentic flow + re-prefill logprob verification.

Design rationale
~~~~~~~~~~~~~~~~
This file needs to run the *exact same* pipeline as the production
``agentic_tool_call.generate``, but insert a verification step between
"collect session records" and "convert records to training samples".
There are two ways to achieve this:

1. **Modify production code** — split ``generate()`` into reusable
   ``_generate_core()`` / ``_finalize()`` helpers and import them here.
2. **Inline the production flow** — copy the pipeline steps verbatim
   from ``agentic_tool_call.generate`` into this file, and insert
   verification in the middle.

We chose option 2 to avoid modifying production code for test purposes.
The tradeoff is a maintenance burden: if ``agentic_tool_call.generate``
changes, this file must be updated in lockstep.

Equivalence guarantee
~~~~~~~~~~~~~~~~~~~~~
Steps 1 and 4 in ``generate()`` below are a *verbatim* copy of the
production ``agentic_tool_call.generate`` (as of the commit that
introduced this test).  Specifically:

- **Step 1** (lines up to ``collect_records``) mirrors production lines
  that create the tracer, load the agent, build metadata, run the agent,
  and collect records.  Every function call, argument, and branch is
  identical.
- **Step 4** (from ``compute_samples_from_openai_records`` onward)
  mirrors the production finalization: sample computation, metadata
  merge, truncation, and multi-sample merge.  Same calls, same order.

Steps 2-3 are *test-only* assertions and the re-prefill verification.
They are read-only checks that do not mutate ``records`` or
``session_metadata``, so they cannot affect the finalization output.

To detect drift, ``grep`` for the functions called in steps 1/4
(``OpenAIEndpointTracer.create``, ``build_chat_request_kwargs``,
``compute_samples_from_openai_records``, ``truncate_samples_by_total_tokens``,
``merge_samples``) — if their signatures change in production, this file
will fail to compile or produce assertion errors.

Verification approach
~~~~~~~~~~~~~~~~~~~~~
After the multi-turn agent session completes, the TITO session server
exposes the full ``accumulated_token_ids`` — the token sequence built
incrementally across all turns.  We send this to ``/generate`` with
``max_new_tokens=0`` for a single prefill pass and compare the resulting
``input_token_logprobs`` against the per-turn ``output_token_logprobs``
from the session's decode phase.  Any token ID mismatch is fatal (TITO
tokenization bug); logprob values must match within a tight tolerance
(prefill vs decode numerical differences).

When ``use_rollout_routing_replay`` is enabled (MoE models), per-turn
``routed_experts`` arrays are also compared against the re-prefill.
"""

import logging
import statistics
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pybase64

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import build_chat_request_kwargs
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
    truncate_samples_by_total_tokens,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

LOGPROB_ABS_TOL = 1e-8  # deterministic inference → prefill and decode must be bit-identical
LOGPROB_WARN_TOL = 0.0  # any nonzero diff is worth logging


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Run production agentic flow with mid-pipeline logprob verification.

    Steps 1 and 4 are a verbatim copy of ``agentic_tool_call.generate``.
    Steps 2-3 are test-only read-only checks inserted between record
    collection and sample finalization.  See module docstring for the
    equivalence rationale.
    """

    # ── Step 1: core agentic flow ──────────────────────────────────────
    # Verbatim from agentic_tool_call.generate — do NOT diverge.
    tracer = await OpenAIEndpointTracer.create(input.args)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    max_seq_len = getattr(input.args, "max_seq_len", None)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}

    agent_metadata = await custom_agent_function(
        base_url=tracer.base_url,
        prompt=input.sample.prompt,
        request_kwargs=build_chat_request_kwargs(input.sampling_params),
        metadata=metadata,
    )

    records, session_metadata = await tracer.collect_records()

    if not records:
        logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)
    # ── End step 1 ─────────────────────────────────────────────────────

    # ── Step 2: test-only session-level precondition checks ────────────
    # These are stricter than production (which silently tolerates missing
    # fields).  We enforce them because the re-prefill comparison below
    # is meaningless without a valid accumulated token sequence.
    mismatch = session_metadata.get("tito_session_mismatch")
    assert mismatch == [], f"tito_session_mismatch is not empty: {mismatch}"

    accumulated = session_metadata.get("accumulated_token_ids")
    assert accumulated and len(accumulated) > 0, "accumulated_token_ids is empty"

    max_trim_tokens = session_metadata.get("max_trim_tokens", 0)

    assert len(records) >= 2, f"Expected at least 2 turns for TITO verification, got {len(records)}"

    # ── Step 3: re-prefill logprob verification (test-only) ────────────
    sglang_url = f"http://{input.args.sglang_router_ip}:{input.args.sglang_router_port}"
    use_r3 = getattr(input.args, "use_rollout_routing_replay", False)

    await _verify_logprobs_via_reprefill(
        sglang_url,
        records,
        accumulated,
        max_trim_tokens=max_trim_tokens,
        use_r3=use_r3,
    )

    logger.info(
        "Logprob equivalence verified: %d turns, %d accumulated tokens",
        len(records),
        len(accumulated),
    )

    # ── Step 4: finalize ───────────────────────────────────────────────
    # Verbatim from agentic_tool_call.generate — do NOT diverge.
    samples = compute_samples_from_openai_records(
        input.args,
        input.sample,
        records,
        input.state.tokenizer,
        accumulated_token_ids=accumulated,
        max_trim_tokens=max_trim_tokens,
    )

    for s in samples:
        s.metadata.update(agent_metadata or {})

    if max_seq_len is not None:
        samples = truncate_samples_by_total_tokens(samples, max_seq_len, input.state.tokenizer)

    if not samples:
        logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
        samples.metadata.update(session_metadata)
    else:
        samples[-1].metadata.update(session_metadata)
    return GenerateFnOutput(samples=samples)
    # ── End step 4 ─────────────────────────────────────────────────────


# Reuse the same CLI arguments as agentic_tool_call so that the
# framework registers --custom-agent-function-path, --max-seq-len, etc.
generate.add_arguments = None  # set below after import


def _init_add_arguments():
    from miles.rollout.generate_hub.agentic_tool_call import generate as _base_generate

    generate.add_arguments = _base_generate.add_arguments


_init_add_arguments()


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _match_output_tokens(
    output_ids: list[int],
    accumulated_token_ids: list[int],
    cursor: int,
) -> int:
    """Count how many leading output tokens match accumulated[cursor:].

    Shared by logprob and routed-expert verification.  Mirrors the
    greedy-match loop in ``compute_samples_from_openai_records``
    (openai_endpoint_utils.py) — intentionally duplicated here to avoid
    modifying production code for test purposes.
    """
    matched = 0
    for j in range(len(output_ids)):
        idx = cursor + j
        if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
            matched += 1
        else:
            break
    return matched


async def _verify_logprobs_via_reprefill(
    sglang_url: str,
    records: list,
    accumulated_token_ids: list[int],
    max_trim_tokens: int,
    use_r3: bool,
) -> None:
    """Re-prefill the full accumulated sequence and compare logprobs.

    Sends ``accumulated_token_ids`` to ``/generate`` with
    ``max_new_tokens=0, return_logprob=True``.  Compares the resulting
    ``input_token_logprobs`` (single prefill pass) against per-turn
    ``output_token_logprobs`` (incremental decode) from session records.
    """
    first_prompt_len = len(records[0].response["choices"][0]["prompt_token_ids"])

    # ── A: send re-prefill request ──
    payload = {
        "input_ids": accumulated_token_ids,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": first_prompt_len,
    }
    if use_r3:
        payload["return_routed_experts"] = True

    reprefill_resp = await post(f"{sglang_url}/generate", payload)
    reprefill_logprobs = reprefill_resp["meta_info"]["input_token_logprobs"]

    expected_len = len(accumulated_token_ids) - first_prompt_len
    assert len(reprefill_logprobs) == expected_len, (
        f"Re-prefill returned {len(reprefill_logprobs)} input_token_logprobs, "
        f"expected {expected_len} (accumulated={len(accumulated_token_ids)}, "
        f"first_prompt={first_prompt_len})"
    )

    # ── B: walk records and compare per-turn ──
    all_diffs: list[float] = []
    # Per-turn (cursor, matched) pairs — reused by R3 verification to
    # avoid recomputing the same greedy match.
    turn_matches: list[tuple[int, int]] = []

    for i, record in enumerate(records):
        is_last = i == len(records) - 1
        choice = record.response["choices"][0]
        prompt_ids = choice["prompt_token_ids"]
        session_output_logprobs = choice["meta_info"]["output_token_logprobs"]
        output_ids = [t[1] for t in session_output_logprobs]

        if not output_ids:
            logger.warning("Turn %d: no output tokens, skipping", i)
            turn_matches.append((len(prompt_ids), 0))
            continue

        cursor = len(prompt_ids)
        matched = _match_output_tokens(output_ids, accumulated_token_ids, cursor)
        turn_matches.append((cursor, matched))

        trim_count = len(output_ids) - matched
        allowed = 0 if is_last else max_trim_tokens
        assert trim_count <= allowed, f"Turn {i}: trim_count {trim_count} exceeds allowed={allowed}"

        # Compare matched tokens against re-prefill
        turn_reprefill_start = cursor - first_prompt_len
        mismatches = []
        warnings = []

        for j in range(matched):
            rp_entry = reprefill_logprobs[turn_reprefill_start + j]  # (logprob, token_id, text)
            sp_entry = session_output_logprobs[j]  # (logprob, token_id)

            rp_tid = rp_entry[1]
            sp_tid = sp_entry[1]
            assert rp_tid == sp_tid, f"Turn {i}, token {j}: token_id mismatch — reprefill={rp_tid} vs session={sp_tid}"

            rp_lp = rp_entry[0]
            sp_lp = sp_entry[0]
            if rp_lp is None or sp_lp is None:
                continue

            diff = abs(rp_lp - sp_lp)
            all_diffs.append(diff)

            if diff > LOGPROB_ABS_TOL:
                mismatches.append(f"  token {j}: prefill={rp_lp:.8f} decode={sp_lp:.8f} diff={diff:.4f}")
            elif diff > LOGPROB_WARN_TOL:
                warnings.append(f"  token {j}: prefill={rp_lp:.8f} decode={sp_lp:.8f} diff={diff:.4f}")

        if warnings:
            logger.warning(
                "Turn %d: %d tokens with diff > %.4f (but within tolerance):\n%s",
                i,
                len(warnings),
                LOGPROB_WARN_TOL,
                "\n".join(warnings[:10]),
            )

        assert (
            not mismatches
        ), f"Turn {i}: {len(mismatches)} logprob differences exceed " f"tolerance {LOGPROB_ABS_TOL}:\n" + "\n".join(
            mismatches
        )

        logger.info("Turn %d: verified %d output tokens (trimmed %d)", i, matched, trim_count)

    # ── C: R3 routed-expert comparison ──
    if use_r3:
        _verify_routed_experts(records, reprefill_resp, accumulated_token_ids, first_prompt_len, turn_matches)

    # ── D: summary statistics ──
    if all_diffs:
        sorted_diffs = sorted(all_diffs)
        p99_idx = min(int(len(sorted_diffs) * 0.99), len(sorted_diffs) - 1)
        logger.info(
            "Logprob diff stats: mean=%.6f, max=%.6f, p99=%.6f, count=%d",
            statistics.mean(all_diffs),
            max(all_diffs),
            sorted_diffs[p99_idx],
            len(all_diffs),
        )


def _verify_routed_experts(
    records: list,
    reprefill_resp: dict,
    accumulated_token_ids: list[int],
    first_prompt_len: int,
    turn_matches: list[tuple[int, int]],
) -> None:
    """Compare per-turn routed_experts from session decode vs re-prefill.

    Reuses ``turn_matches`` (cursor, matched) computed by the logprob
    verification pass to avoid recomputing the greedy token match.
    """
    reprefill_re_b64 = reprefill_resp["meta_info"].get("routed_experts")
    if reprefill_re_b64 is None:
        logger.warning("Re-prefill response missing routed_experts, skipping R3 check")
        return

    reprefill_re_flat = np.frombuffer(pybase64.b64decode(reprefill_re_b64.encode("ascii")), dtype=np.int32)

    for i, record in enumerate(records):
        choice = record.response["choices"][0]
        prompt_ids = choice["prompt_token_ids"]
        output_logprobs = choice["meta_info"]["output_token_logprobs"]
        output_ids = [t[1] for t in output_logprobs]

        session_re_b64 = choice["meta_info"].get("routed_experts")
        if session_re_b64 is None:
            logger.warning("Turn %d: session missing routed_experts, skipping", i)
            continue

        cursor, matched = turn_matches[i]
        if matched == 0:
            continue

        session_re = np.frombuffer(pybase64.b64decode(session_re_b64.encode("ascii")), dtype=np.int32)

        # routed_experts shape: [total_tokens - 1, num_layers, top_k].
        # Entry k corresponds to the routing decision for token k+1.
        total_tokens = len(prompt_ids) + len(output_ids)
        if total_tokens <= 1 or len(session_re) == 0:
            continue

        per_token_size = len(session_re) // (total_tokens - 1)

        # Session output starts at position P (=len(prompt_ids)),
        # so its expert entries start at index (P-1) in the flat array.
        session_start = (len(prompt_ids) - 1) * per_token_size
        session_slice = session_re[session_start : session_start + matched * per_token_size]

        # Re-prefill covers all of accumulated_token_ids.
        # Token at accumulated[cursor] → expert entry at index (cursor-1).
        rp_offset = (cursor - 1) * per_token_size
        rp_slice = reprefill_re_flat[rp_offset : rp_offset + matched * per_token_size]

        np.testing.assert_array_equal(
            session_slice,
            rp_slice,
            err_msg=f"Turn {i}: routed_experts mismatch",
        )
        logger.info("Turn %d: routed_experts match (%d entries)", i, len(session_slice))
