"""Rollout top-k logprobs: the sampler's candidate log-probs recorded for each generated token.

``--rollout-top-logprobs-num`` sets the width of ``Sample.rollout_topk_token_ids`` /
``rollout_topk_log_probs``; it records the sampling distribution and does not change it.
``arguments.py`` derives ``rollout_sampling_logprobs_mode`` from the rollout filters: ``selected`` reads
``output_top_logprobs``, the top-K of the full-vocabulary distribution (unfiltered sampling);
``support`` reads ``output_token_sampling_logprobs`` over the whole realized support (filtered
sampling). ``Sample.rollout_sampling_mask`` is the separate support record of sampling-support replay.
"""

import math
from collections.abc import Mapping
from typing import Any


def validate_rollout_topk_logprobs_args(args: Any) -> None:
    """Reject rollout top-k logprobs settings whose recorded candidates cannot match the sampler."""
    k = args.rollout_top_logprobs_num
    if k < 0:
        raise ValueError(f"--rollout-top-logprobs-num must be non-negative, got {k}")
    if not k:
        return
    if args.rollout_sampling_logprobs_mode == "support" and k < args.rollout_top_k:
        raise ValueError(
            "Filtered rollout sampling requires --rollout-top-logprobs-num >= --rollout-top-k "
            "so the recorded candidates can hold the whole sampling support"
        )
    opd_student_top_k = args.use_opd and args.opd_log_prob_top_k > 0 and args.opd_top_k_strategy != "only-teacher"
    if opd_student_top_k:
        raise ValueError("--rollout-top-logprobs-num cannot be combined with OPD student top-k log-probs")


def validate_rollout_topk_logprobs_sampling(
    sampling: Mapping[str, Any], *, temperature: float, candidate_count: int
) -> None:
    """Require per-call sampling whose distribution the recorded candidates describe.

    Top-p/top-k bounds are checked by the rollout args and sampling-support replay;
    this adds what recorded candidates need on top of them.
    """
    if sampling.get("temperature", temperature) != temperature or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Rollout top-k logprobs collection requires the same positive rollout temperature on every generation call"
        )
    if sampling.get("top_k", -1) > candidate_count:
        raise ValueError(
            "Rollout top-k logprobs collection requires top_k <= --rollout-top-logprobs-num to hold the whole support"
        )
    if sampling.get("min_p", 0.0) != 0.0:
        raise ValueError("Rollout top-k logprobs collection requires min_p=0.0")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "custom_logit_processor", "logit_bias"):
        if sampling.get(key):
            raise ValueError(f"Rollout top-k logprobs collection does not support constrained/custom sampling ({key})")
    response_format = sampling.get("response_format")
    if response_format and (not isinstance(response_format, Mapping) or response_format.get("type", "text") != "text"):
        raise ValueError("Rollout top-k logprobs collection does not support constrained response_format")
    tool_choice = sampling.get("tool_choice", "auto")
    if tool_choice not in (None, "auto", "none"):
        raise ValueError("Rollout top-k logprobs collection does not support constrained tool_choice")
    if tool_choice != "none" and any(tool.get("function", {}).get("strict") for tool in sampling.get("tools") or []):
        raise ValueError("Rollout top-k logprobs collection does not support strict tool schemas")
