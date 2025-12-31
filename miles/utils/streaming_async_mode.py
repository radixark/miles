import logging

logger = logging.getLogger(__name__)


def validate_streaming_async_args(args) -> None:
    # In this fork, PipelineRL is the streaming-async semantics; keep them coupled.
    if getattr(args, "pipeline_rl", False) and not getattr(args, "streaming_async", False):
        raise ValueError("--pipeline-rl requires --streaming-async")

    if not getattr(args, "streaming_async", False):
        return

    if not getattr(args, "pipeline_rl", False):
        raise ValueError("--streaming-async requires --pipeline-rl")

    if getattr(args, "use_miles_router", False):
        raise ValueError("--streaming-async does not support MilesRouter")

    if args.balance_data and args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]:
        raise ValueError(
            "--streaming-async group-as-atom requires not splitting prompt groups across DP partitions; "
            "disable --balance-data for this advantage estimator."
        )

    if args.pipeline_weight_update_interval < 1:
        raise ValueError("--pipeline-weight-update-interval must be >= 1")

    if args.pipeline_max_weight_lag < 0:
        raise ValueError("--pipeline-max-weight-lag must be >= 0")

    if not args.use_tis:
        logger.warning("--streaming-async is enabled; consider adding --use-tis for off-policy tolerance.")
