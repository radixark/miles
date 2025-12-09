import random
import bisect
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MinimumFibonacciRepresentation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3539
    prompt_template = \
r"""Define Fibonacci numbers as the sequence: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ... You can represent any positive integer by adding or subtracting Fibonacci numbers. For example:
- 10 = 5 + 5 → uses 2 Fibonacci numbers
- 19 = 21 - 2 → uses 2 Fibonacci numbers
- 17 = 13 + 5 - 1 → uses 3 Fibonacci numbers
- 1070 = 987 + 89 - 5 - 1 → uses 4 Fibonacci numbers

Please compute the minimum number of Fibonacci numbers needed (added or subtracted) to represent the number {K}. Output a single integer — the minimum number of Fibonacci numbers used."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Multiplication_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 10, "MAX_K should be greater than or equal to 10"

        K = self.parameter["K"] = random.randint(4, MAX_K)


        # Build the Fibonacci-like sequence up to just above maxK
        F = [1, 2]
        while F[-1] <= K:
            F.append(F[-2] + F[-1])
        # Now F[-1] > maxK, F[-2] <= maxK

        RES = 0
        n = K
        while n:
            RES += 1
            # Find first F element > n
            idx = bisect.bisect_right(F, n)
            larger = F[idx]
            smaller = F[idx - 1]
            # Move n toward zero by the minimal step
            n = min(larger - n, n - smaller)
        self.parameter["reference_answer"] = RES
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(K = self.parameter["K"])


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