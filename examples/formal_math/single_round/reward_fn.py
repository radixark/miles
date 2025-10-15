from typing import Optional

from kimina_wrapper import KiminaServerAndClientCluster
from kimina_client import SnippetStatus

_TIMEOUT = 60


class RewardFn:
    def __init__(self):
        self._verifier = KiminaServerAndClientCluster()

    async def __call__(self, args, sample, **kwargs):
        try:
            code, code_error_cat = _extract_code(prompt=sample.prompt, response=sample.response)
            if code is None:
                return dict(reward_value=0.0, error_cat=code_error_cat)

            resp = await self._verifier.check(codes=[dict(code=code, custom_id="dummy_id")], timeout=_TIMEOUT)
            result = _single(resp.results)
            analysis = result.analyze()

            return dict(
                reward_value=float(analysis.status == SnippetStatus.valid),
                error_cat=None if is_valid_no_sorry else "LEAN_VERIFY_FAILED",
                lean_result=result,
            )
        except Exception as e:
            logger.warning(f"Error in reward_value model: {e=} {sample.prompt=} {sample.response=}")
            traceback.print_exc()
            return dict(reward_value=0.0, error_cat="PYTHON_ERROR", error_details=str(e))


def _single(arr):
    assert len(arr) == 1, f"{arr=}"
    return arr[0]


_REWARD_FN: Optional[RewardFn] = None


async def reward_fn(*args, **kwargs):
    global _REWARD_FN
    if _REWARD_FN is None:
        _REWARD_FN = RewardFn()
    return _REWARD_FN(*args, **kwargs)
