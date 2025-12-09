import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MatrixPermutation_BothDiagonalOne_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a square matrix of size {N} × {N}, where each element is either `0` or `1`. This matrix is 0-indexed.

Please find:
- a permutation of the row indices: a[0], ..., a[{N_minus_1}] (a reordering of `0` to `{N_minus_1}`),
- a permutation of the column indices: b[0], ..., b[{N_minus_1}] (a reordering of `0` to `{N_minus_1}`),
- such that after applying these permutations to the rows and columns of matrix A (i.e., the element at position (i, j) becomes A[a[i]][b[j]]), **both diagonals of the resulting matrix contain only `1`s** — that is, all positions where `i = j` (main diagonal) and `i + j = {N_minus_1}` (anti-diagonal).

Matrix A is given as follows:
{A}

**Output Format:** Output two lines:
- The first line contains the row permutation: a[0] a[1] ... a[{N_minus_1}]
- The second line contains the column permutation: b[0] b[1] ... b[{N_minus_1}]
(Use spaces to separate adjacent integers. Do **not** include backticks or quotes.)"""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the MatrixPermutation_BothDiagonalOne_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter
        N = self.parameter["N"]
        assert N >= 2, "N must be at least 2."

        one_probability = random.random() / 4.0
        A = self.parameter["A"] = [[1 if random.random() < one_probability else 0 for _ in range(N)] for _ in range(N)]

        row_permutation = list(range(N))
        random.shuffle(row_permutation)
        column_permutation = list(range(N))
        random.shuffle(column_permutation)
        for i in range(N) :
            A[row_permutation[i]][column_permutation[i]] = 1
        for i in range(N) :
            A[row_permutation[i]][column_permutation[N - 1 - i]] = 1
        self.parameter["reference_answer"] = " ".join(map(str, row_permutation)) + "\n" + " ".join(map(str, column_permutation))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = "\n".join("".join(map(str, row)) for row in self.parameter["A"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                permutations = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        permutations.append(list(map(int, line.split())))
                if len(permutations) == 2 :
                    return permutations[0], permutations[1]
                else :
                    return None
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            row_permutation, column_permutation = processed_result
            if not (len(row_permutation) == self.parameter["N"] and set(row_permutation) == set(range(self.parameter["N"]))) :
                return self.rewards["invalid_solution"]
            if not (len(column_permutation) == self.parameter["N"] and set(column_permutation) == set(range(self.parameter["N"]))) :
                return self.rewards["invalid_solution"]
            B = [[self.parameter["A"][row_permutation[i]][column_permutation[j]] for j in range(self.parameter["N"])] for i in range(self.parameter["N"])]

            satisfied, total = 0, 0
            for i in range(self.parameter["N"]) :
                for j in range(self.parameter["N"]) :
                    if i == j or i + j == self.parameter["N"] - 1 :
                        total += 1
                        satisfied += B[i][j]
            assert satisfied <= total, "satisfied must be less than or equal to total"
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / total) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == total)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]