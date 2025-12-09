import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CongruentEquation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1082
    prompt_template = \
r"""Find the **smallest positive integer solution** `x` to the following congruence equation:

`{A} * x ≡ 1 (mod {B})`

Output Format:
Your final answer should be a single positive integer representing the smallest solution `x`.
Example: `17` (do **NOT** include the backticks or quotes).
"""
    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs):
        """
        Initialize the CongruentEquation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_A_B" in self.parameter, "MAX_A_B is required in parameter"
        MAX_A_B = self.parameter["MAX_A_B"]
        assert MAX_A_B >= 2, "MAX_A_B should be greater than or equal to 1"

        while True :
            A = self.parameter["A"] = random.randint(1, MAX_A_B)
            B = self.parameter["B"] = random.randint(2, MAX_A_B)
            
            def exgcd(a, b) :
                if b == 0 :
                    return a, 1, 0
                d, x1, y1 = exgcd(b, a % b)
                x = y1
                y = x1 - (a // b) * y1
                return d, x, y
            d, x, y = exgcd(A, B)

            if d == 1 :
                x = (x % B + B) % B
                assert x > 0, "x should be positive, but got {}".format(x)
                assert A * x % B == 1, "A * x % B should be equal to 1, but got {} != {}".format(A * x % B, 1)
                self.parameter["reference_answer"] = x
                break
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            A = self.parameter["A"],
            B = self.parameter["B"],
        )
    

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
            if processed_result <= 0 :
                return self.rewards["wrong_format"]
            if self.parameter["A"] * processed_result % self.parameter["B"] != 1 :
                return self.rewards["invalid_answer"]
            
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                assert processed_result > self.parameter["reference_answer"], "processed_result should be greater than reference_answer, but got {} <= {}".format(processed_result, self.parameter["reference_answer"])
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]