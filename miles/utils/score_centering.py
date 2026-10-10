"""Configuration contracts for score centering (arXiv:2609.20807)."""

import math
import os
from argparse import Namespace

from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sampling


def validate_score_centering_args(args: Namespace) -> None:
    if getattr(args, "loss_type", None) != "score_centering":
        return
    if args.rollout_top_logprobs_num <= 0:
        raise ValueError("Score centering requires a positive --rollout-top-logprobs-num")
    if not math.isfinite(args.score_centering_tis_clip) or args.score_centering_tis_clip <= 0:
        raise ValueError("--score-centering-tis-clip must be finite and positive")
    low, high = args.score_centering_mis_low, args.score_centering_mis_high
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    validate_rollout_topk_logprobs_sampling(
        {"top_k": args.rollout_top_k},
        temperature=args.rollout_temperature,
        candidate_count=args.rollout_top_logprobs_num,
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
        "custom_convert_samples_to_train_data_path": "custom converters skip rollout top-k logprobs validation",
    }
    for option, reason in incompatible.items():
        if getattr(args, option, None):
            raise ValueError(f"Score centering is incompatible with --{option.replace('_', '-')}: {reason}")
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
        raise ValueError("Score centering requires SGLANG_RETURN_ORIGINAL_LOGPROB=0 on rollout servers")
