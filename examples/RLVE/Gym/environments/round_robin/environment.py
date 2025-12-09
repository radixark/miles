import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class RoundRobin_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Please construct an {N} × {N} matrix, where each element is either 0, 1, or 2. Denote the matrix as A (0-indexed), and it must satisfy the following conditions:
1. A[i][i] = 0 for all i.
2. For all i ≠ j (0 ≤ i, j < {N}), A[i][j] + A[j][i] = 2 (i.e., one of the following holds: A[i][j] = 0 and A[j][i] = 2; A[i][j] = 2 and A[j][i] = 0; or A[i][j] = A[j][i] = 1).
3. Define W[i] = 3 × (number of positions j where A[i][j] = 2) + 1 × (number of positions j where A[i][j] = 1). The final values of W[0], ..., W[{N_minus_1}] must be exactly: {W}

**Output Format:** Output {N} lines, each containing {N} digits (0, 1, or 2) with no separators. The i-th line should represent A[i][0], A[i][1], ..., A[i][{N_minus_1}]."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the RoundRobin_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_beta": rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        tie_probability = random.random()
        A = [[None] * N for _ in range(N)]
        self.parameter["W"] = W = [0] * N
        for i in range(N) :
            for j in range(N) :
                if i == j :
                    A[i][j] = 0
                    continue
                if i < j :
                    if random.random() < tie_probability :
                        A[i][j] = 1
                    else :
                        A[i][j] = random.choice([0, 2])
                else :
                    A[i][j] = 2 - A[j][i]
                W[i] += 3 * (A[i][j] == 2) + 1 * (A[i][j] == 1)
        self.parameter["reference_answer"] = "\n".join("".join(map(str, row)) for row in A)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            W = " ".join("W[{}]={}".format(i, Wi) for i, Wi in enumerate(self.parameter["W"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(line.strip())
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            A = processed_result
            if len(A) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if any(len(row) != self.parameter["N"] for row in A) :
                return self.rewards["wrong_format"]
            if any(any(c not in "012" for c in row) for row in A) :
                return self.rewards["wrong_format"]
            
            W = [0] * self.parameter["N"]
            for i in range(self.parameter["N"]) :
                for j in range(self.parameter["N"]) :
                    if i == j :
                        if A[i][j] != "0" :
                            return self.rewards["invalid_solution"]
                    else :
                        if int(A[i][j]) + int(A[j][i]) != 2 :
                            return self.rewards["invalid_solution"]
                        assert (A[i][j] == "0" and A[j][i] == "2") or (A[i][j] == "2" and A[j][i] == "0") or (A[i][j] == A[j][i] == "1")
                    W[i] += 3 * (A[i][j] == "2") + 1 * (A[i][j] == "1")
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["W"], W)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["W"] == W)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]