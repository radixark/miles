import itertools
import logging

from miles.utils.multi_lora import is_multi_lora_enabled

logger = logging.getLogger(__name__)


def postprocess_rollout_data(args, data, train_parallel_config):
    metadata = {}

    # Multi-LoRA: record group boundaries (heterogeneous per-adapter group sizes)
    # and lift the collection loop's batch-level step decision out of sample metadata,
    # both before flattening.
    if is_multi_lora_enabled(args) and isinstance(data[0], list):
        metadata["prompt_group_sizes"] = [_nested_sample_count(group) for group in data]
        head = _first_sample(data[0])
        metadata["step_slots"] = list(head.metadata.pop("step_slots", []))
        metadata["step_adapter_names"] = list(head.metadata.pop("step_adapter_names", []))

    # flatten the data if it is a list of lists
    while isinstance(data[0], list):
        data = list(itertools.chain.from_iterable(data))

    if not args.disable_rollout_trim_samples:
        global_batch_size = args.global_batch_size
        if args.use_dynamic_global_batch_size:
            logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
            dynamic_global_batch_size = _compute_dynamic_global_batch_size(
                args, train_parallel_config=train_parallel_config, num_samples=len(data)
            )
            metadata["dynamic_global_batch_size"] = dynamic_global_batch_size
            global_batch_size = dynamic_global_batch_size

        if len(data) % global_batch_size != 0:
            trim_len = (len(data) // global_batch_size) * global_batch_size
            if trim_len == 0:
                raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
            origin_data_length = len(data)
            data = data[:trim_len]
            logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
        logger.info(f"Final collected {len(data)} samples from rollout to train")

    return data, metadata


def _first_sample(group):
    return _first_sample(group[0]) if isinstance(group[0], list) else group[0]


def _nested_sample_count(group) -> int:
    if not isinstance(group, list):
        return 1
    return sum(_nested_sample_count(item) for item in group)


def _compute_dynamic_global_batch_size(args, train_parallel_config, num_samples: int) -> int:
    """Calculate dynamic global_batch_size to ensure only one training step.

    Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
    This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
    """
    dp_size = train_parallel_config["dp_size"]
    original_gbs = args.global_batch_size

    if is_multi_lora_enabled(args):
        # Batches take groups in multiples of each adapter's
        # min_groups_per_dp_split, so this holds by construction; a violation
        # means a generate fn's group shape broke the invariant.
        if num_samples % dp_size != 0:
            raise ValueError(
                f"Multi-LoRA batch of {num_samples} samples is not divisible by dp_size={dp_size}; "
                "the min_groups_per_dp_split invariant was violated (variable-size generate fn output?)"
            )
        return num_samples

    # Round down to a multiple of dp_size to ensure only one training step
    dynamic_gbs = (num_samples // dp_size) * dp_size

    if dynamic_gbs == 0:
        # Too few samples, use at least dp_size
        dynamic_gbs = dp_size
        logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

    # Calculate how many samples will be discarded
    wasted = num_samples - dynamic_gbs

    if dynamic_gbs != original_gbs or wasted > 0:
        logger.info(
            f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
            f"(num_samples={num_samples}, dp_size={dp_size}, "
            f"num_steps=1, wasted={wasted})"
        )

    return dynamic_gbs
