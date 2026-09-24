from miles.utils.args.schema import A, Arg, BaseConfig


class RolloutBufferConfig(BaseConfig):
    rollout_buffer_url: A[str | None, Arg(help="URL for the rollout buffer")] = None
    fetch_trajectory_retry_times: A[
        int,
        Arg(help="Number of times to retry fetching trajectory, -1 means unlimited retry"),
    ] = -1
    min_batch_collection_ratio: A[float, Arg(help="Minimum batch collection ratio")] = 1
    rollout_task_type: A[str, Arg()] = "math"
    data_pad_size_multiplier: A[
        int,
        Arg(help="Multiplier for data padding size in data processing."),
    ] = 128
    disable_rollout_trim_samples: A[
        bool,
        Arg(help="disable trim samples in rollout buffer when converting samples to train data"),
    ] = False
    use_dynamic_global_batch_size: A[
        bool,
        Arg(
            help="enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data"
        ),
    ] = False


class RolloutBufferRolloutOnlyConfig(BaseConfig):
    loss_mask_type: A[
        str,
        Arg(
            choices=["qwen", "qwen3", "distill_qwen"],
            help="Loss mask type",
        ),
    ] = "qwen"
    rollout_sample_filter_path: A[
        str | None,
        Arg(
            help=(
                "Path to the rollout sample filter function. "
                "This function determines whether a sample will participate in loss calculation. "
                "The function is called as `fn(args, data)` where `data` is `list[list[Sample]]` "
                "(grouped by n_samples_per_prompt), and should return None. "
                "To exclude a sample from the loss, set `sample.remove_sample = True`. "
                "Note: This attribute does not determine whether the sample participates in advantage normalization."
            )
        ),
    ] = None
    rollout_all_samples_process_path: A[
        str | None,
        Arg(
            help=(
                "Path to the rollout all samples process function that "
                "can process all samples including filtered ones."
            )
        ),
    ] = None
