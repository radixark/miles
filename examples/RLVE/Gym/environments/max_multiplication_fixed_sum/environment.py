import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MaxMultiplicationFixedSum_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4157
    prompt_template = r"""Can you tell me the maximum product of positive integers (not necessarily distinct) whose sum is exactly {N}?"""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MaxMultiplicationFixedSum_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 10, "MAX_N should be greater than or equal to 10"

        N = self.parameter["N"] = random.randint(4, MAX_N)


        n = N
        if n % 3 == 0 :
            ans = 3 ** (int(n / 3))
        if n % 3 == 1 :
            ans = 3 ** (int((n - 4) / 3)) * 4
        if n % 3 == 2 :
            ans = 3 ** (int((n - 2) / 3)) * 2
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])


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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]