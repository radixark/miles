import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MinNoSolutionLinearDiophantineEquation_Environment(VerifiableEnvironment) : # https://www.luogu.com.cn/problem/P3951
    prompt_template = \
r"""Consider the equation {A}x + {B}y = z. Find the largest non-negative integer z ≥ 0 such that the equation has **no** non-negative integer solutions (x, y)."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MinNoSolutionLinearDiophantineEquation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_A_B" in self.parameter, "MAX_A_B is required in parameter"
        MAX_A_B = self.parameter["MAX_A_B"]
        assert MAX_A_B >= 3, "A and B should be greater than or equal to 3"

        while True :
            A = self.parameter["A"] = random.randint(2, MAX_A_B)
            B = self.parameter["B"] = random.randint(2, MAX_A_B)
            if math.gcd(A, B) == 1 :
                break

        # The smallest non-negative integer z such that the equation has no non-negative integer solutions is A * B - A - B.
        self.parameter["reference_answer"] = A * B - A - B
        assert self.parameter["reference_answer"] > 0
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(A = self.parameter["A"], B = self.parameter["B"])


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