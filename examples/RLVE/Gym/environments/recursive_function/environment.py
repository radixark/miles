import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class RecursiveFunction_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Define a function f(m, n) as follows:
1. If m = 0, then f(m, n) = n + 1.
2. If m > 0 and n = 0, then f(m, n) = f(m - 1, 1).
3. If m > 0 and n > 0, then f(m, n) = f(m // 2, f(m // 2, n // 2)) + f(m // 2, f(m // 2, n - 1)). Here, `//` denotes integer division.

Please compute the value of f({M}, {N})
"""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the RecursiveFunction_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_M_N" in self.parameter, "MAX_M_N is required in parameter"
        MAX_M_N = self.parameter["MAX_M_N"]
        assert MAX_M_N >= 1, "MAX_M_N should be greater than or equal to 1"

        M, N = self.parameter["M"], self.parameter["N"] = random.randint(1, MAX_M_N), random.randint(1, MAX_M_N)


        ackermann = dict()
        def ack(m, n) :
            if m == 0 :
                return n + 1
            if (m, n) not in ackermann :
                if n == 0 :
                    ackermann[(m, n)] = ack(m - 1, 1)
                else :
                    ackermann[(m, n)] = ack(m // 2, ack(m // 2, n // 2)) + ack(m // 2, ack(m // 2, n - 1))
            return ackermann[(m, n)]
        self.parameter["reference_answer"] = ack(M, N)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(M = self.parameter["M"], N = self.parameter["N"])


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