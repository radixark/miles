"""Request-scoped policy semantics for the Tinker compatibility API."""

from contextlib import contextmanager


@contextmanager
def external_tinker_policy_flags(args):
    names = ("use_rollout_logprobs", "use_tis", "get_mismatch_metrics", "use_opsm")
    previous = {name: getattr(args, name, False) for name in names}
    args.use_rollout_logprobs = True
    args.use_tis = False
    args.get_mismatch_metrics = False
    args.use_opsm = False
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(args, name, value)
