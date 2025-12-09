import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SurvoPuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} matrix with some cells filled with numbers from `0` to `{NM_minus_1}`, and some cells empty (represented by `-1`). Please fill the empty cells with numbers from `0` to `{NM_minus_1}` such that:
1. Each number from `0` to `{NM_minus_1}` appears **exactly once** in the matrix.
2. The sum of each row (from top to bottom) is: {row_sums}
3. The sum of each column (from left to right) is: {col_sums}

The matrix is given as follows:
{matrix}

**Output Format:** Your final answer should contain {N} lines, each with {M} numbers, separated by spaces. The numbers should represent the completed matrix in **row-major order**, matching the format of the given input."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the SurvoPuzzle_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)

        permutation = list(range(N * M))
        random.shuffle(permutation)

        matrix = self.parameter["matrix"] = [[permutation[i * M + j] for j in range(M)] for i in range(N)]
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in matrix)
        self.parameter["row_sums"] = [sum(row) for row in matrix]
        self.parameter["col_sums"] = [sum(matrix[i][j] for i in range(N)) for j in range(M)]

        assert "sparsity" in self.parameter, "sparsity is required in parameter"
        sparsity = self.parameter["sparsity"]
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        empty_cells = random.sample(range(N * M), max(1, int(N * M * sparsity)))
        for cell in empty_cells :
            row, column = divmod(cell, M)
            matrix[row][column] = -1
    

    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            NM_minus_1 = N * M - 1,
            matrix = "\n".join(" ".join(map(str, row)) for row in self.parameter["matrix"]),
            row_sums = " ".join(map(str, self.parameter["row_sums"])),
            col_sums = " ".join(map(str, self.parameter["col_sums"])),
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

            N, M = self.parameter["N"], self.parameter["M"]
            solution = processed_result
            
            if len(solution) != N or any(len(row) != M for row in solution) :
                return self.rewards["wrong_format"]

            if set(value for row in solution for value in row) != set(range(N * M)) :
                return self.rewards["invalid_solution"]
            if any(original_value != -1 and original_value != solution_value for original_row, solution_row in zip(self.parameter["matrix"], solution) for original_value, solution_value in zip(original_row, solution_row)) :
                return self.rewards["invalid_solution"]
            
            row_sums = [sum(row) for row in solution]
            col_sums = [sum(solution[i][j] for i in range(N)) for j in range(M)]

            satisfied = sum(int(answer == gold) for answer, gold in zip(row_sums, self.parameter["row_sums"])) + \
                        sum(int(answer == gold) for answer, gold in zip(col_sums, self.parameter["col_sums"]))
            assert satisfied <= N + M, "satisfied should not exceed N + M"

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (N + M)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == (N + M))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]