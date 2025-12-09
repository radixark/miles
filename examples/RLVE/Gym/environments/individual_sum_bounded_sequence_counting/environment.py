import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class IndividualSumBounded_SequenceCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3228
    prompt_template = \
r"""Count the number of sequences X[1], ..., X[{K}] such that:
- X[1] ≥ 1
- For all i in [2, {K}]: 1 ≤ X[i] ≤ {M}
- The total sum X[1] + X[2] + ... + X[{K}] ≤ {N}

Output the count modulo {MOD}."""

    def __init__(self,
                 max_MOD : int = 1000000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the IndividualSumBounded_SequenceCounting_Environment instance.
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
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 2, "MAX_N should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N)
        K = self.parameter["K"] = int(2 ** random.uniform(1.0, math.log2(N)))
        M = self.parameter["M"] = random.randint(1, (N - 1) // (K - 1))
        assert K >= 2, "K should be at least 2"
        assert 1 + M * (K - 1) <= N, "N should be at least 1 + M * (K - 1)"

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        pow1 = pow(M, K-1, MOD)
        pow2 = pow(M, K-2, MOD)
        # term1 = N * M^(K-1) mod MOD
        term1 = (N % MOD) * pow1 % MOD
        # x = M*(M+1)/2 mod MOD
        x = (M * (M + 1) // 2) % MOD
        # term2 = (K-1) * x * M^(K-2) mod MOD
        term2 = ( (K-1) % MOD ) * x % MOD * pow2 % MOD
        # answer = term1 - term2  (mod MOD)
        ans = (term1 - term2) % MOD
        self.parameter["reference_answer"] = ans


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"], M = self.parameter["M"], MOD = self.parameter["MOD"])


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