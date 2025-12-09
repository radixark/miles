import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MatrixPermutationEquivalence_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given two matrices A and B of size {N} × {M}, where each element is either `0` or `1`. Both matrices are 0-indexed.

Please find:
- a permutation of the row indices `a[0], ..., a[{N_minus_1}]` (a reordering of `0` to `{N_minus_1}`), and
- a permutation of the column indices `b[0], ..., b[{M_minus_1}]` (a reordering of `0` to `{M_minus_1}`),
- such that after permuting the rows and columns of matrix A accordingly, the resulting matrix matches B. Formally, for all `0 ≤ i < {N}` and `0 ≤ j < {M}`, it must hold that A[a[i]][b[j]] = B[i][j].

A is given as follows:
{A}

B is given as follows:
{B}

**Output Format:** Output two lines:
- The first line contains the row permutation: `a[0] ... a[{N_minus_1}]`
- The second line contains the column permutation: `b[0] ... b[{M_minus_1}]`
(Use spaces to separate the adjacent integers on the same line. Do **not** include backticks or quotes.)"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the MatrixPermutationEquivalence_Environment instance.
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
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)
        one_probability = random.random()
        A = self.parameter["A"] = [[1 if random.random() < one_probability else 0 for _ in range(M)] for _ in range(N)]

        row_permutation = list(range(N))
        random.shuffle(row_permutation)
        column_permutation = list(range(M))
        random.shuffle(column_permutation)

        self.parameter["B"] = [[A[row_permutation[i]][column_permutation[j]] for j in range(M)] for i in range(N)]
        self.parameter["reference_answer"] = " ".join(map(str, row_permutation)) + "\n" + " ".join(map(str, column_permutation))
    

    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            N_minus_1 = N - 1,
            M_minus_1 = M - 1,
            A = "\n".join("".join(map(str, row)) for row in self.parameter["A"]),
            B = "\n".join("".join(map(str, row)) for row in self.parameter["B"]),
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
            if not (len(column_permutation) == self.parameter["M"] and set(column_permutation) == set(range(self.parameter["M"]))) :
                return self.rewards["invalid_solution"]
            B = [[self.parameter["A"][row_permutation[i]][column_permutation[j]] for j in range(self.parameter["M"])] for i in range(self.parameter["N"])]

            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(sum(answer == gold for answer, gold in zip(answer_row, gold_row)) for answer_row, gold_row in zip(B, self.parameter["B"])) / (self.parameter["N"] * self.parameter["M"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (B == self.parameter["B"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]