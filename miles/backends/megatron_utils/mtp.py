"""Read Megatron's MTP loss with the same reduction for direct and API training."""


def consume_mtp_loss(*, num_microbatches: int, calculate_per_token_loss: bool) -> float | None:
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    MTPLossLoggingHelper.reduce_loss_in_tracker()
    tracker = MTPLossLoggingHelper.tracker
    loss = tracker.get("values", tracker.get("loss_values"))
    if loss is None:
        return None
    value = (loss * (1.0 if calculate_per_token_loss else 1 / num_microbatches)).item()
    MTPLossLoggingHelper.clean_loss_in_tracker()
    return value
