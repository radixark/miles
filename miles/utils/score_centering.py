"""Configuration contracts for score centering (arXiv:2609.20807)."""

import math
import os
from argparse import Namespace
from collections.abc import Mapping
from typing import Any


def score_centering_top_k(args: Namespace) -> int:
    return args.score_centering_top_k if getattr(args, "loss_type", None) == "score_centering" else 0


def validate_score_centering_sampling(
    sampling: Mapping[str, Any], *, temperature: float, candidate_count: int
) -> None:
    """Require a bounded support covered by the recorded candidates.

    Filtered generation requests SGLang's post-filter support probabilities.
    The candidate count must cover the complete realized support.
    """
    if sampling.get("temperature", temperature) != temperature or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Score centering requires the same positive rollout temperature on every generation call")
    top_p = sampling.get("top_p", 1.0)
    top_k = sampling.get("top_k", -1)
    if not 0 < top_p <= 1 or (top_k != -1 and not 0 < top_k <= candidate_count):
        raise ValueError("Score centering requires top_p in (0, 1] and top_k=-1 or 1..score_centering_top_k")
    if top_p < 1 and top_k == -1:
        raise ValueError("Score centering requires positive top_k with top_p filtering to bound the support")
    if sampling.get("min_p", 0.0) != 0.0:
        raise ValueError("Score centering requires min_p=0.0")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "custom_logit_processor", "logit_bias"):
        if sampling.get(key):
            raise ValueError(f"Score centering does not support constrained/custom sampling ({key})")
    response_format = sampling.get("response_format")
    if response_format and (not isinstance(response_format, Mapping) or response_format.get("type", "text") != "text"):
        raise ValueError("Score centering does not support constrained response_format")
    tool_choice = sampling.get("tool_choice", "auto")
    if tool_choice not in (None, "auto", "none"):
        raise ValueError("Score centering does not support constrained tool_choice")
    if tool_choice != "none" and any(tool.get("function", {}).get("strict") for tool in sampling.get("tools") or []):
        raise ValueError("Score centering does not support strict tool schemas")


def validate_score_centering_args(args: Namespace) -> None:
    if getattr(args, "loss_type", None) != "score_centering":
        return
    if args.score_centering_top_k <= 0:
        raise ValueError("--score-centering-top-k must be positive")
    if not math.isfinite(args.score_centering_tis_clip) or args.score_centering_tis_clip <= 0:
        raise ValueError("--score-centering-tis-clip must be finite and positive")
    low, high = args.score_centering_mis_low, args.score_centering_mis_high
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    validate_score_centering_sampling(
        {"top_p": args.rollout_top_p, "top_k": args.rollout_top_k},
        temperature=args.rollout_temperature,
        candidate_count=args.score_centering_top_k,
    )
    if args.advantage_estimator != "grpo":
        raise ValueError("Score centering currently supports --advantage-estimator grpo (group-centered rewards)")
    incompatible = {
        "use_tis": "use --score-centering-is instead",
        "custom_tis_function_path": "only the built-in score-centering TIS/MIS weights are supported",
        "use_opsm": "sequence masking changes the score-centering estimator",
        "true_on_policy_mode": "score centering uses float32 probability arithmetic",
        "recompute_logprobs_via_prefill": "sampler probabilities must be recorded at generation time",
        "sglang_speculative_algorithm": "speculative candidate-logprob semantics are not verified",
        "custom_pg_loss_reducer_function_path": "use the standard token/sample reducer",
        "multi_lora": "per-sample Tinker losses bypass the score-centering loss",
        "use_opd": "distillation composition is not supported",
    }
    for option, reason in incompatible.items():
        if getattr(args, option, None):
            raise ValueError(f"Score centering is incompatible with --{option.replace('_', '-')}: {reason}")
    if (
        args.score_centering_top_k > 20
        and args.rollout_top_p == 1.0
        and args.rollout_top_k == -1
        and getattr(args, "use_session_server", None)
        and not getattr(args, "use_miles_router", False)
    ):
        raise ValueError(
            "Score-centering session rollouts with more than 20 candidates require --use-miles-router: "
            "the SGLang Rust router caps OpenAI top_logprobs at 20"
        )
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
        raise ValueError("Score centering requires SGLANG_RETURN_ORIGINAL_LOGPROB=0 on rollout servers")
