import asyncio
import logging
import re
from types import SimpleNamespace
from typing import Optional, Tuple

from kimina_client import SnippetStatus

try:
    import kimina_wrapper
except ImportError:
    from . import kimina_wrapper

logger = logging.getLogger(__name__)

_TIMEOUT = 60


class RewardFn:
    def __init__(self):
        self._verifier = kimina_wrapper.KiminaServerAndClientCluster()

    async def __call__(self, args, sample, **kwargs):
        try:
            code, code_error_cat = _extract_code(prompt=sample.prompt, response=sample.response)
            if code is None:
                return dict(reward_value=0.0, error_cat=code_error_cat)

            resp = await self._verifier.check(snips=code, timeout=_TIMEOUT, show_progress=False)
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
    return await _REWARD_FN(*args, **kwargs)


if __name__ == "__main__":
    # Run this UT with:
    # python examples/formal_math/single_round/reward_fn.py

    # test example is from stoney0062/Leanabell-Prover-Traindata-SFT
    test_prompt = """
    Hello

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- A pirate is counting his loot in base 6. He has $4532_6$ dollars worth of silver, $1254_6$ dollars worth of pearls, and $654_6$ dollars worth of exotic spices. What is the total dollar amount of his loot? Express your answer in base 10. Show that it is 1636.-/
theorem pirate_loot_total (silver pearls spices : ℕ) (h_silver : silver = 4 * 6 ^ 3 + 5 * 6 ^ 2 + 3 * 6 + 2)
(h_pears : pearls = 1 * 6 ^ 3 + 2 * 6 ^ 2 + 5 * 6 + 4) (h_spices : spices = 6 * 6 ^ 2 + 5 * 6 + 4) :
silver + pearls + spices = 1636 := by
  sorry
```
    """
    test_response = """
```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- A pirate is counting his loot in base 6. He has $4532_6$ dollars worth of silver, $1254_6$ dollars worth of pearls, and $654_6$ dollars worth of exotic spices. What is the total dollar amount of his loot? Express your answer in base 10. Show that it is 1636.-/
theorem pirate_loot_total (silver pearls spices : ℕ) (h_silver : silver = 4 * 6 ^ 3 + 5 * 6 ^ 2 + 3 * 6 + 2)
(h_pears : pearls = 1 * 6 ^ 3 + 2 * 6 ^ 2 + 5 * 6 + 4) (h_spices : spices = 6 * 6 ^ 2 + 5 * 6 + 4) :
silver + pearls + spices = 1636 := by
  rw [h_silver, h_pears, h_spices]
  norm_num
  <;> linarith
```
    """

    import ray

    ray.init()
    output = asyncio.run(reward_fn(None, SimpleNamespace(prompt=test_prompt, response=test_response)))
    print(f"{output=}")
    assert output['reward_value'] == 1.0
