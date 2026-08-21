import copy
import itertools
import logging

from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def postprocess_rollout_data(args, data, train_parallel_config, pad_to_dp: bool = False):
    metadata = {}

    validate_compact_rollout_ids(data)

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

    if pad_to_dp and (dp_size := (train_parallel_config or {}).get("dp_size")):
        data = _pad_samples_to_dp(data, dp_size)

    # Compact rollouts must not be trimmed by sample count; the schedule drops
    # whole trailing rollouts instead.
    is_compact = any(s.rollout_id is not None for s in data)

    if not args.disable_rollout_trim_samples and not is_compact:
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


def validate_compact_rollout_ids(node, depth=0):
    """Require compact leaves (``list[Sample]`` at depth >= 2, >1 sibling) to
    share a non-None ``rollout_id``; default rollout shapes skip validation."""
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [s.rollout_id for s in node]
            missing = [i for i, r in enumerate(rids) if r is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
                f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
                "reducer can aggregate them as one rollout instead of N."
            )
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        validate_compact_rollout_ids(item, depth + 1)


def _first_sample(group):
    return _first_sample(group[0]) if isinstance(group[0], list) else group[0]


def _nested_sample_count(group) -> int:
    if not isinstance(group, list):
        return 1
    return sum(_nested_sample_count(item) for item in group)


def _pad_samples_to_dp(data: list[Sample], dp_size: int) -> list[Sample]:
    deficit = -len(data) % dp_size
    if deficit == 0:
        return data

    donor = data[-1]
    padded = list(data)
    for _ in range(deficit):
        pad = copy.deepcopy(donor)
        pad.index = -1
        pad.rollout_id = None
        pad.loss_mask = [0] * pad.response_length
        padded.append(pad)
    logger.info(f"Padded rollout batch from {len(data)} to {len(padded)} samples for DP alignment")
    return padded


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
