from Gym.environment import VerifiableEnvironment
from typing import Optional, List
import random


class SpiralMatrix_Environment(VerifiableEnvironment):
    prompt_template = \
r"""You are given a 2D integer matrix of size {M} x {N}:
{matrix}

Return all elements of the matrix in a clockwise spiral order, starting from the top-left corner. More precisely:
- Start from the top-left corner and move right until you reach the right edge.
- Then, move down until you reach the bottom-right corner.
- Then, move left until you reach the bottom-left corner.
- Then, move up until you reach the top-right corner.
- Continue this inward spiral traversal until all elements have been visited exactly once.

**Output Format:**
Your final answer should be a single line of {MN} integers separated by **spaces**.

---

**Example 1**

You are given an integer matrix of size 3 x 3:
1 2 3
4 5 6
7 8 9

The output is (do **NOT** include backticks or quotes — use the format below exactly):
```
1 2 3 6 9 8 7 4 5
```

**Example 2**

You are given an integer matrix of size 3 x 4:
1 2 3 4
5 6 7 8
9 10 11 12

The output is (do **NOT** include backticks or quotes — use the format below exactly):
```
1 2 3 4 8 12 11 10 9 5 6 7
```
---
"""

    def __init__(
        self,
        wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
        **kwargs
    ):
        """
        Initialize the SpiralMatrixProblem instance.
        """
        super().__init__(**kwargs)
        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

    def _generate(self) -> None:
        assert "MAX_M_N" in self.parameter, "MAX_M_N is required in parameter"
        MAX_M_N = self.parameter["MAX_M_N"]
        self.parameter["M"] = M = random.randint(2, MAX_M_N)
        self.parameter["N"] = N = random.randint(2, MAX_M_N)

        self.matrix = [[random.randint(1, M * N) for _ in range(N)] for _ in range(M)]
        self.parameter["matrix"] = self.matrix
        self.parameter["gold_answer"] = self._compute_spiral(self.matrix)
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))

    def _prompt_generate(self) -> str:
        return self.prompt_template.format(
            M=self.parameter["M"],
            N=self.parameter["N"],
            MN=self.parameter["M"] * self.parameter["N"],
            matrix="\n".join(" ".join(map(str, row)) for row in self.parameter["matrix"]),
        )

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format

    def _compute_spiral(self, matrix: List[List[int]]) -> List[int]:
        res = []
        if not matrix:
            return res

        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1

        while top <= bottom and left <= right:
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            top += 1

            for i in range(top, bottom + 1):
                res.append(matrix[i][right])
            right -= 1

            if top <= bottom:
                for i in range(right, left - 1, -1):
                    res.append(matrix[bottom][i])
                bottom -= 1

            if left <= right:
                for i in range(bottom, top - 1, -1):
                    res.append(matrix[i][left])
                left += 1

        return res

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["M"] * self.parameter["N"] :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / (self.parameter["M"] * self.parameter["N"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]