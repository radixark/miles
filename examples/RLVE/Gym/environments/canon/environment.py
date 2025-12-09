import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Canon_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3214
    prompt_template = \
r"""Let S be the set of integers from 1 to {N} ({N} integers in total).

Please count the number of sequences T[1], ..., T[{M}] such that:
- Each T[i] is a **non-empty subset** of S.
- For each integer x in [1, {N}], the total number of subsets T[i] that contain x is an **even number** (including 0).
- T[1], ..., T[{M}] are **distinct** subsets.

**Output Format:** Output a single integer — the number of valid sequences T, modulo {MOD}."""


    def __init__(self,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Canon_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)
        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        # tot = (2^N mod MOD) - 1
        tot = pow(2, N, MOD) - 1

        # Precompute A[i] = (tot) * (tot - 1) * ... * (tot - (i - 1)) mod MOD
        A = [0] * (M + 1)
        A[0] = 1
        for i in range(1, M + 1):
            # multiply by (tot - (i - 1)), ensure non-negative before mod
            A[i] = A[i - 1] * ((tot - (i - 1)) % MOD) % MOD

        # f[i] will count, up to multiplying by i!, the number of valid sequences of i distinct subsets
        f = [0] * (M + 1)
        f[0] = 1
        # f[1] stays 0 (no way to have one non-empty subset and all pitches even)
        for i in range(2, M + 1):
            # start with all ways to pick (i-1) distinct subsets
            val = A[i - 1]
            # subtract those where the i-th subset repeated some previous pattern
            val = (val - f[i - 1]) % MOD
            # subtract configurations where a pitch appears an odd number of times due to overlaps
            # the correction term is f[i-2] * (i-1) * (tot - (i-2))
            correction = f[i - 2] * (i - 1) % MOD * ((tot - (i - 2)) % MOD) % MOD
            val = (val - correction) % MOD
            f[i] = val

        self.parameter["reference_answer"] = f[M]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"], MOD = self.parameter["MOD"])


    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]