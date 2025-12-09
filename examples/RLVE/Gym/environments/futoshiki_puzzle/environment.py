import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class FutoshikiPuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {N} matrix. Some cells are already filled with integers in the range [0, {N_minus_1}], and the rest are empty (denoted by `-1`). Please fill the empty cells with integers in the same range such that:
- Each **row** and each **column** contains all integers from `0` to `{N_minus_1}` **exactly once**.
- The following **inequality constraints** between cells are satisfied (use `c[i][j]` to denote the cell at row `i`, column `j`, 0-indexed):  
{inequalities}

The original matrix is as follows:  
{matrix}

**Output Format:** Your final answer should contain {N} lines, each with {N} integers separated by spaces. Each line represents a row of the completed matrix, matching the format of the input."""

    def __init__(self,
                 inequality_constraint_num_multiple : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the FutoshikiPuzzle_Environment instance.
        """
        super().__init__(**kwargs)

        self.inequality_constraint_num_multiple = inequality_constraint_num_multiple

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        permutation_row, permutation_col = list(range(N)), list(range(N))
        random.shuffle(permutation_row)
        random.shuffle(permutation_col)

        self.parameter["matrix"] = matrix = [[(permutation_row[i] + permutation_col[j]) % N for j in range(N)] for i in range(N)]
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in matrix)

        all_inequalities = []
        for x1 in range(N) :
            for y1 in range(N) :
                for x2 in range(N) :
                    for y2 in range(N) :
                        if matrix[x1][y1] < matrix[x2][y2] :
                            all_inequalities.append((x1, y1, x2, y2))
        self.parameter["inequalities"] = random.sample(all_inequalities, k = random.randint(1, min(len(all_inequalities), self.inequality_constraint_num_multiple * N)))

        assert "sparsity" in self.parameter, "sparsity is required in parameter"
        sparsity = self.parameter["sparsity"]
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        empty_cells = random.sample(range(N * N), max(1, int(N * N * sparsity)))
        for cell in empty_cells :
            row, column = divmod(cell, N)
            matrix[row][column] = -1
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            matrix = "\n".join(" ".join(map(str, row)) for row in self.parameter["matrix"]),
            inequalities = "\n".join("c[{}][{}] < c[{}][{}]".format(x1, y1, x2, y2) for x1, y1, x2, y2 in self.parameter["inequalities"]),
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

            N = self.parameter["N"]
            solution = processed_result
            
            if len(solution) != N or any(len(row) != N for row in solution) :
                return self.rewards["wrong_format"]
            
            if not all(set(row) == set(range(N)) for row in solution) :
                return self.rewards["invalid_solution"]
            if not all(set(solution[i][j] for i in range(N)) == set(range(N)) for j in range(N)) :
                return self.rewards["invalid_solution"]
            
            if any(original_value != -1 and original_value != value for original_row, row in zip(self.parameter["matrix"], solution) for original_value, value in zip(original_row, row)) :
                return self.rewards["invalid_solution"]
            
            satisfied = sum(int(solution[x1][y1] < solution[x2][y2]) for x1, y1, x2, y2 in self.parameter["inequalities"])
            assert satisfied <= len(self.parameter["inequalities"]), "satisfied should not exceed the number of inequalities"

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / len(self.parameter["inequalities"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (satisfied == len(self.parameter["inequalities"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]