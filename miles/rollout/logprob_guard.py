import math

from miles.utils.types import Sample


def guard_rollout_log_probs(samples: list[Sample], loss_masks: list[list[int]]) -> None:
    """Reject non-finite rollout log probabilities on trainable tokens."""
    nan_count = 0
    inf_count = 0
    valid_token_count = 0
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
            if not is_active:
                continue
            valid_token_count += 1
            if math.isfinite(log_prob):
                continue
            row_is_bad = True
            if math.isnan(log_prob):
                nan_count += 1
            else:
                inf_count += 1
        if row_is_bad:
            bad_rows.append(row)

    non_finite_count = nan_count + inf_count
    if non_finite_count:
        raise ValueError(
            f"Non-finite rollout_log_probs detected: {non_finite_count} bad tokens "
            f"(nan={nan_count}, inf={inf_count}) out of {valid_token_count} valid tokens, "
            f"in {len(bad_rows)}/{len(samples)} sequences "
            f"(first bad row indices: {bad_rows[:16]})"
        )
