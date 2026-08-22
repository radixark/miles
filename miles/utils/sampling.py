from argparse import Namespace


def uses_sampling_support_truncation(*, top_p: float, top_k: int, min_p: float = 0.0) -> bool:
    return top_p < 1.0 or top_k > 0 or min_p > 0.0


def sampling_mask_replay_enabled(args: Namespace) -> bool:
    return uses_sampling_support_truncation(
        top_p=float(getattr(args, "rollout_top_p", 1.0)),
        top_k=int(getattr(args, "rollout_top_k", -1)),
    )
