"""Request-scoped policy semantics for the Tinker compatibility API."""

from contextlib import contextmanager


@contextmanager
def external_tinker_policy_flags(args, loss_fn_config=None):
    names = (
        "use_rollout_logprobs",
        "use_tis",
        "get_mismatch_metrics",
        "use_opsm",
        "eps_clip",
        "eps_clip_high",
    )
    previous = {name: getattr(args, name, False) for name in names}
    clip_low = clip_high = None
    if loss_fn_config:
        clip_low = float(loss_fn_config.get("clip_low_threshold", 1 - args.eps_clip))
        clip_high = float(loss_fn_config.get("clip_high_threshold", 1 + args.eps_clip_high))
        if not 0 <= clip_low <= 1 <= clip_high:
            raise ValueError(
                "Tinker PPO clipping thresholds must satisfy "
                "0 <= clip_low_threshold <= 1 <= clip_high_threshold"
            )
    args.use_rollout_logprobs = True
    args.use_tis = False
    args.get_mismatch_metrics = False
    args.use_opsm = False
    if clip_low is not None and clip_high is not None:
        args.eps_clip = 1 - clip_low
        args.eps_clip_high = clip_high - 1
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(args, name, value)
