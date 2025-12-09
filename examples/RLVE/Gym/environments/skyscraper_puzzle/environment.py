import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SkyscraperPuzzle_Environment(VerifiableEnvironment):
    prompt_template = \
r"""You are given a {N} × {N} grid. Your task is to place a building of height in the range [0, {N_minus_1}] in each cell such that:
- Each **row** and each **column** contains all integer heights from `0` to `{N_minus_1}` **exactly once**.
- A building is **visible from a direction** if there are no taller buildings before it in that direction.

The number of visible buildings is specified as follows:
- From the **left** of each row: {left}
- From the **right** of each row: {right}
- From the **top** of each column: {top}
- From the **bottom** of each column: {bottom}

**Output Format:** Your final answer should contain {N} lines, each with {N} integers (heights), separated by spaces. Each line represents a row of the grid."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SkyscraperPuzzle_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        permutation_row, permutation_col = list(range(N)), list(range(N))
        random.shuffle(permutation_row)
        random.shuffle(permutation_col)

        grid = [[(permutation_row[i] + permutation_col[j]) % N for j in range(N)] for i in range(N)]
        self.parameter["left"] = [sum(int(grid[i][j] == max(grid[i][: j + 1])) for j in range(N)) for i in range(N)]
        self.parameter["right"] = [sum(int(grid[i][j] == max(grid[i][j :])) for j in range(N)) for i in range(N)]

        transposed_grid = [[grid[j][i] for j in range(N)] for i in range(N)]
        self.parameter["top"] = [sum(int(transposed_grid[i][j] == max(transposed_grid[i][: j + 1])) for j in range(N)) for i in range(N)]
        self.parameter["bottom"] = [sum(int(transposed_grid[i][j] == max(transposed_grid[i][j :])) for j in range(N)) for i in range(N)]

        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in grid)
    

    def _prompt_generate(self) -> str :
        N =  self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            left = " ".join(map(str, self.parameter["left"])),
            right = " ".join(map(str, self.parameter["right"])),
            top = " ".join(map(str, self.parameter["top"])),
            bottom = " ".join(map(str, self.parameter["bottom"])),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                grid = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        grid.append(list(map(int, line.split())))
                return grid
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
            
            if len(solution) != N :
                return self.rewards["wrong_format"]
            if not all(len(row) == N for row in solution) :
                return self.rewards["wrong_format"]
            
            if not all(set(row) == set(range(N)) for row in solution) :
                return self.rewards["invalid_solution"]
            if not all(set(solution[i][j] for i in range(N)) == set(range(N)) for j in range(N)) :
                return self.rewards["invalid_solution"]
            
            left = [sum(int(solution[i][j] == max(solution[i][: j + 1])) for j in range(N)) for i in range(N)]
            right = [sum(int(solution[i][j] == max(solution[i][j :])) for j in range(N)) for i in range(N)]

            transposed_solution = [[solution[j][i] for j in range(N)] for i in range(N)]
            top = [sum(int(transposed_solution[i][j] == max(transposed_solution[i][: j + 1])) for j in range(N)) for i in range(N)]
            bottom = [sum(int(transposed_solution[i][j] == max(transposed_solution[i][j :])) for j in range(N)) for i in range(N)]

            satisfied = sum(int(answer == gold) for answer, gold in zip(left, self.parameter["left"])) + \
                        sum(int(answer == gold) for answer, gold in zip(right, self.parameter["right"])) + \
                        sum(int(answer == gold) for answer, gold in zip(top, self.parameter["top"])) + \
                        sum(int(answer == gold) for answer, gold in zip(bottom, self.parameter["bottom"]))
            assert satisfied <= 4 * N, "satisfied should not exceed 4 * N"

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (4 * N)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (satisfied == (4 * N))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]