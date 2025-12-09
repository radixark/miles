import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class StirlingSecond_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1287
    prompt_template = \
r"""There are {R} distinct boxes and {N} distinct balls. Count the number of ways to place all {N} balls into the boxes such that **no box is empty**. Two arrangements are different if **at least one ball** is placed into a different box. Output the result modulo {MOD}."""
    MOD = 10**9 + 7

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the StirlingSecond_Environment instance.
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

        assert "MAX_R" in self.parameter, "MAX_R is required in parameter"
        MAX_R = self.parameter["MAX_R"]
        assert MAX_R >= 2, "MAX_R should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N)
        R = self.parameter["R"] = random.randint(2, min(N, MAX_R))
        MOD = self.MOD

        ans = 0
        c = 1
        for k in range(R) :
            term = c * pow(R - k, N, MOD) % MOD
            if k & 1 :
                ans = ((ans - term) % MOD + MOD) % MOD
            else :
                ans = (ans + term) % MOD
            c = c * (R - k) // (k + 1)
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], R = self.parameter["R"], MOD = self.MOD)


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
            if not (0 <= processed_result < self.MOD) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]