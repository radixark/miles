from typing import Optional

from lean_verifier import LeanVerifier


class RewardFn:
    def __init__(self):
        self._lean_verifier = LeanVerifier()

    def __call__(self, args, sample, **kwargs):
        TODO


_REWARD_FN: Optional[RewardFn] = None


async def reward_fn(*args, **kwargs):
    global _REWARD_FN
    if _REWARD_FN is None:
        _REWARD_FN = RewardFn()
    return _REWARD_FN(*args, **kwargs)
