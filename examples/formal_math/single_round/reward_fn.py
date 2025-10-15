from typing import Optional

from lean_verifier import LeanVerifier

_TIMEOUT = 60

class RewardFn:
    def __init__(self):
        self._lean_verifier = LeanVerifier()

    async def __call__(self, args, sample, **kwargs):
        try:
            code, code_error_cat = _extract_code(prompt=sample.prompt, response=sample.response)
            if code is None:
                return dict(reward_value=0.0, error_cat=code_error_cat)

            response = await self._lean_verifier.check(codes=[dict(code=code, custom_id="dummy_id")], timeout=_TIMEOUT)
            assert len(response["results"]) == 1, f"{response=}"
            raw_result = response["results"][0]
            parsed_result = parse_client_response(raw_result)
            is_valid_no_sorry = parsed_result["is_valid_no_sorry"]

            return dict(
                reward_value=float(is_valid_no_sorry),
                error_cat=None if is_valid_no_sorry else "LEAN_VERIFY_FAILED",
                lean_result=raw_result,
            )
        except Exception as e:
            logger.warning(f"Error in reward_value model: {e=} {sample.prompt=} {sample.response=}")
            traceback.print_exc()
            return dict(reward_value=0.0, error_cat="PYTHON_ERROR", error_details=str(e))


_REWARD_FN: Optional[RewardFn] = None


async def reward_fn(*args, **kwargs):
    global _REWARD_FN
    if _REWARD_FN is None:
        _REWARD_FN = RewardFn()
    return _REWARD_FN(*args, **kwargs)
