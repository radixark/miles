import logging
import re
import traceback
from typing import Optional, Tuple

from kimina_wrapper import KiminaServerAndClientCluster
from kimina_client import SnippetStatus

logger = logging.getLogger(__name__)

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
            is_valid = analysis.status == SnippetStatus.valid

            return dict(
                reward_value=float(is_valid),
                error_cat=None if is_valid else f"lean_{analysis.status.value}",
                lean_result=result.model_dump(),
            )
        except Exception as e:
            logger.warning(f"Error in RewardFn: {e=} {sample.prompt=} {sample.response=}")
            return dict(reward_value=0.0, error_cat="PYTHON_ERROR", error_details=str(e))


def _single(arr):
    assert len(arr) == 1, f"{arr=}"
    return arr[0]


def _extract_code(prompt: str, response: str) -> Tuple[Optional[str], Optional[str]]:
    question_code = _extract_last_full_code_block(prompt)
    assert question_code is not None

    response_code = _extract_last_full_code_block(response)
    if response_code is None:
        return None, "no_code"

    if _canonicalize_question(question_code) != _canonicalize_question(response_code):
        return None, "question_changed"

    return response_code, None


# hacky
def _canonicalize_question(question_code: str):
    x = question_code
    x = x.replace("\n", "").replace(" ", "")

    try:
        def_symbol_index = x.index(":=")
    except ValueError:
        def_symbol_index = len(x)

    x = x[:def_symbol_index]
    return x


def _extract_last_full_code_block(text):
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1] if matches else None


_REWARD_FN: Optional[RewardFn] = None


async def reward_fn(*args, **kwargs):
    global _REWARD_FN
    if _REWARD_FN is None:
        _REWARD_FN = RewardFn()
    return _REWARD_FN(*args, **kwargs)
