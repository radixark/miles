import math

from miles.utils.types import Sample


def _invalid_log_prob_kind(log_prob: object) -> str | None:
    """Classify non-finite values and their JSON null representation."""
    if log_prob is None:
        return "null"
    if isinstance(log_prob, bool):
        return "invalid_type"

    try:
        if math.isfinite(log_prob):
            return None
        return "nan" if math.isnan(log_prob) else "inf"
    except (TypeError, ValueError, OverflowError):
        return "invalid_type"


def guard_rollout_log_probs(samples: list[Sample], loss_masks: list[list[int]]) -> None:
    """Reject malformed values and non-finite trainable rollout log probabilities."""
    invalid_counts = {"active_nan": 0, "active_inf": 0, "null": 0, "invalid_type": 0}
    total_token_count = 0
    active_token_count = 0
    bad_rows = []

    for row, (sample, loss_mask) in enumerate(zip(samples, loss_masks, strict=True)):
        log_probs = sample.rollout_log_probs
        if log_probs is None:
            continue
        if len(log_probs) != len(loss_mask):
            raise ValueError(
                f"rollout_log_probs length ({len(log_probs)}) != loss_mask length ({len(loss_mask)}) " f"at row {row}"
            )

        row_is_bad = False
        for log_prob, is_active in zip(log_probs, loss_mask, strict=True):
            total_token_count += 1
            if is_active:
                active_token_count += 1
            invalid_kind = _invalid_log_prob_kind(log_prob)
            if invalid_kind is None:
                continue
            if invalid_kind in ("nan", "inf"):
                if not is_active:
                    continue
                invalid_kind = f"active_{invalid_kind}"
            row_is_bad = True
            invalid_counts[invalid_kind] += 1
        if row_is_bad:
            bad_rows.append(row)

    invalid_count = sum(invalid_counts.values())
    if invalid_count:
        raise ValueError(
            f"Invalid rollout_log_probs detected: {invalid_count} bad tokens "
            f"(active_nan={invalid_counts['active_nan']}, active_inf={invalid_counts['active_inf']}, "
            f"null={invalid_counts['null']}, invalid_type={invalid_counts['invalid_type']}) "
            f"across {total_token_count} total tokens ({active_token_count} active), "
            f"in {len(bad_rows)}/{len(samples)} sequences "
            f"(first bad row indices: {bad_rows[:16]})"
        )
