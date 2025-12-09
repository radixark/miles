import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MatrixPooling_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a matrix of size {N} × {M}. Perform a **max pooling** operation with a kernel size of {K} × {K}. In max pooling, each output cell contains the **maximum value** in the corresponding {K} × {K} submatrix of the input.

The matrix is:
{matrix}

**Output Format:** Your output should contain {N} - {K} + 1 lines, each with {M} - {K} + 1 integers separated by **spaces**. Each integer represents the maximum value in the respective pooling region."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MatrixPooling_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(3, MAX_N_M), random.randint(3, MAX_N_M)
        K = self.parameter["K"] = random.randint(2, min(N, M) - 1)

        matrix = self.parameter["matrix"] = [[random.randint(0, N * M) for _ in range(M)] for _ in range(N)]

        gold_answer = self.parameter["gold_answer"] = [[max(matrix[i + di][j + dj] for di in range(K) for dj in range(K)) for j in range(M - K + 1)] for i in range(N - K + 1)]
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in gold_answer)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            K = self.parameter["K"],
            matrix = "\n".join(" ".join(map(str, row)) for row in self.parameter["matrix"]),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(list(map(int, line.split())))
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            pool = processed_result
            if len(pool) != self.parameter["N"] - self.parameter["K"] + 1 :
                return self.rewards["wrong_format"]
            if not all(len(row) == self.parameter["M"] - self.parameter["K"] + 1 for row in pool) :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(sum(answer == gold for answer, gold in zip(answer_row, gold_row)) for answer_row, gold_row in zip(pool, self.parameter["gold_answer"])) / ((self.parameter["N"] - self.parameter["K"] + 1) * (self.parameter["M"] - self.parameter["K"] + 1))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return pool == self.parameter["gold_answer"]
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]