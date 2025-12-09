import random
import math
from typing import Optional
from Gym.environment import VerifiableEnvironment


class ClearSymmetry_Environment(VerifiableEnvironment):
    prompt_template = \
r"""Consider some square matrix A with side n consisting of zeros and ones. There are n rows numbered from 1 to n from top to bottom and n columns numbered from 1 to n from left to right in this matrix. We'll denote the element of the matrix which is located at the intersection of the i-row and the j-th column as A(i, j).

Let's call matrix A clear if no two cells containing ones have a common side.
Let's call matrix A symmetrical if it matches the matrices formed from it by a horizontal and/or a vertical reflection. Formally, for each pair (i, j) (1 ≤ i, j ≤ n) both of the following conditions must be met: A(i, j) = A(n - i + 1, j) and A(i, j) = A(i, n - j + 1).
Let's define the sharpness of matrix A as the number of ones in it.

Given integer x = {x}, your task is to find the smallest positive integer n such that there exists a clear symmetrical matrix A with side n and sharpness x.
Please output only the integer n in your response without any other text.
"""

    def __init__(self,
                 wrong_format: float = -1.0, correct_solution: float = 1.0, incorrect_solution: float = 0.0,
                 **kwargs):
        """
        Initialize the ClearSymmetry_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "correct_solution": correct_solution,
            "incorrect_solution": incorrect_solution,
        }

    def _generate(self) -> None:
        assert "MAX_X" in self.parameter, "MAX_X is required in parameter"
        MAX_X = self.parameter["MAX_X"]
        assert MAX_X >= 1, "MAX_X should be greater than or equal to 1"

        self.parameter["x"] = random.randint(1, MAX_X)
        x = self.parameter["x"]

        # Compute the reference answer using the provided solution, source: https://codeforces.com/contest/201/submission/163120300
        def find_smallest_positive_integer(n: int) -> int:
            if n == 3:
                return 5
            n = math.ceil(math.sqrt(2*n-1))
            return n + 1-n%2

        self.parameter["reference_answer"] = find_smallest_positive_integer(x)

    def _prompt_generate(self) -> str:
        return self.prompt_template.format(x=self.parameter["x"])

    def _process(self, answer: Optional[str]) -> Optional[int]:
        if answer is not None:
            answer = answer.strip()
            try:
                int_answer = int(answer)
                return int_answer
            except ValueError:
                return None
        else:
            return None

    def scorer(self, output: str) -> float:
        processed_result = self.processor(output)
        if processed_result is not None:
            if processed_result == self.parameter["reference_answer"]:
                return self.rewards["correct_solution"]
            else:
                return self.rewards["incorrect_solution"]
        else:
            return self.rewards["wrong_format"] 