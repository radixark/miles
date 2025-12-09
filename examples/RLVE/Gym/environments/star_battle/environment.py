import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class StarBattle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} grid. Each cell contains either `X` or `.`. Please select some `.` cells to fill with `*` such that:
1. Each **row** contains **exactly one** `*`.
2. Each **column** contains **no more than one** `*`.
3. No two `*` cells are adjacent (including diagonals — i.e., no two `*`s share an 8-neighbor relationship).

The grid is given in **row-major order**, with each row represented as a string of `X` and `.`:
{grid}

**Output Format:** Output {N} lines, each containing {M} characters. Each character should be `X`, `.`, or `*`. The output must match the format of the input (i.e., one row per line, no separators), indicating the final state of the grid after placing the `*` cells."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, wrong_solution : float = 0.0, correct_solution : float = 1.0,
                 **kwargs) :
        """
        Initialize the StarBattle_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "wrong_solution" : wrong_solution,
            "correct_solution" : correct_solution,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        while True :
            N = self.parameter["N"] = random.randint(2, MAX_N_M)
            M = self.parameter["M"] = random.randint(max(3, N), MAX_N_M)
            self.parameter["grid"] = grid = [["."] * M for _ in range(N)]
            permutation = random.sample(range(M), N)
            if any(abs(a - b) <= 1 for a, b in zip(permutation, permutation[1 :])) :
                continue
            for row, col in enumerate(permutation) :
                grid[row][col] = "*"
            break
        
        assert "sparsity" in self.parameter, "sparsity is required in parameter"
        sparsity = self.parameter["sparsity"]
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        empty_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "."]
        for i, j in random.sample(empty_cells, max(1, int(len(empty_cells) * sparsity))) :
            grid[i][j] = "X"
        self.parameter["reference_answer"] = "\n".join("".join(row) for row in grid)

        for i in range(N) :
            for j in range(M) :
                if grid[i][j] == "*" :
                    grid[i][j] = "."
                assert grid[i][j] in "X.", "grid should only contain 'X' or '.'"
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            grid = "\n".join("".join(row) for row in self.parameter["grid"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                grid = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        grid.append(line.strip())
                return grid
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
            if not all(c in "X.*" for row in solution for c in row) :
                return self.rewards["wrong_format"]
            
            for row, original_row in zip(solution, self.parameter["grid"]) :
                for cell, original_cell in zip(row, original_row) :
                    if original_cell == "X" and cell != "X" :
                        return self.rewards["invalid_solution"]
                    if original_cell == "." and cell not in ".*" :
                        return self.rewards["invalid_solution"]
            
            if any(row.count("*") != 1 for row in solution) :
                return self.rewards["wrong_solution"]
            if any(col.count("*") > 1 for col in zip(*solution)) :
                return self.rewards["wrong_solution"]
            
            for i in range(N) :
                for j in range(M) :
                    if solution[i][j] == "*" :
                        for di in (-1, 0, +1) :
                            for dj in (-1, 0, +1) :
                                if (di != 0 or dj != 0) and 0 <= i + di < N and 0 <= j + dj < M :
                                    if solution[i + di][j + dj] == "*" :
                                        return self.rewards["wrong_solution"]
            return self.rewards["correct_solution"]
        else :
            return self.rewards["wrong_format"]