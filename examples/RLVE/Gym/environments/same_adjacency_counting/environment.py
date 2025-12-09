import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SameAdjacencyCounting_Environment(VerifiableEnvironment) : # Submitted to https://www.luogu.com.cn/problem/P3197
    prompt_template = \
r"""Count the number of length-{N} sequences using integers from `1` to `{M}` such that **at least one pair of adjacent elements is equal**. Output the result modulo {MOD}."""

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SameAdjacencyCounting_Environment instance.
        """
        super().__init__(**kwargs)

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

        assert "MAX_M" in self.parameter, "MAX_M is required in parameter"
        MAX_M = self.parameter["MAX_M"]
        assert MAX_M >= 2, "MAX_M should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N)
        M = self.parameter["M"] = random.randint(2, MAX_M)
        MOD = self.parameter["MOD"] = random.randint(M, 2 * M)

        self.parameter["reference_answer"] = (pow(M, N, MOD) - M * pow(M - 1, N - 1, MOD) + MOD) % MOD
    

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