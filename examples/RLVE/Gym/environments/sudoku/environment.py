import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Sudoku_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Solve a Sudoku puzzle of size ({N} × {M}) × ({M} × {N}) = {NM} × {NM}. Each number is in the range from 1 to {NM}, and empty cells are represented by 0. Here is the input grid:
{sudoku}

Rules of Sudoku:
1. Each **row** must contain all digits from 1 to {NM}, without repetition.
2. Each **column** must contain all digits from 1 to {NM}, without repetition.
3. The grid is divided into {M} × {N} **subgrids**, where each subgrid is of size {N} × {M} (i.e., each subgrid has {N} rows and {M} columns). Each subgrid must also contain all digits from 1 to {NM}, without repetition.

**Output Format:**
Your final answer should contain {NM} lines, each with {NM} numbers, separated by spaces. The numbers should represent the completed Sudoku grid in **row-major order**, matching the format of the given input — that is, the first number on the first line is the top-left cell of the Sudoku. Example (do **NOT** include the backticks or quotes, and this is NOT a valid Sudoku):
```
{output_example}
```"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, wrong_solution : float = 0.0, correct_solution : float = 1.0,
                 **kwargs) :
        """
        Initialize the Sudoku_Environment instance.
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
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)
        NM = self.parameter["NM"] = N * M


        base = [[(M * (row % N) + row // N + column) % NM + 1 for column in range(NM)] for row in range(NM)]

        perm = list(range(1, NM + 1))
        random.shuffle(perm)
        grid = [[perm[base[row][column] - 1] for column in range(NM)] for row in range(NM)]

        def shuffle_groups(data, group_size) :
            G = len(data) // group_size
            for g in range(G) :
                start = g * group_size
                slice_ = data[start : start + group_size]
                random.shuffle(slice_)
                data[start : start+group_size] = slice_
            groups = [data[g * group_size:(g + 1) * group_size] for g in range(G)]
            random.shuffle(groups)
            data[:] = [row for group in groups for row in group]

        shuffle_groups(grid, N)
        grid_t = list(map(list, zip(*grid)))
        shuffle_groups(grid_t, M)
        grid = list(map(list, zip(*grid_t)))

        if random.choice([True, False]) :
            grid = list(map(list, zip(*grid)))
            N, M = M, N
            self.parameter["N"], self.parameter["M"] = N, M

        
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in grid)


        assert "sparsity" in self.parameter, "sparsity is required in parameter"
        sparsity = self.parameter["sparsity"]
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        empty_cells = random.sample(range(NM * NM), max(1, int(NM * NM * sparsity)))
        for cell in empty_cells :
            row, column = divmod(cell, NM)
            grid[row][column] = 0
        self.parameter["sudoku"] = grid

    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            NM = N * M,
            sudoku = "\n".join(" ".join(map(str, row)) for row in self.parameter["sudoku"]),
            output_example = "\n".join(" ".join(map(str, range(1, N * M + 1))) for _ in range(N * M))
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

            N, M = self.parameter["N"], self.parameter["M"]
            solution = processed_result
            
            if len(solution) != N * M :
                return self.rewards["wrong_format"]
            for row in solution :
                if len(row) != N * M :
                    return self.rewards["wrong_format"]
            
            for solution_row, sudoku_row in zip(solution, self.parameter["sudoku"]) :
                for solution_cell, sudoku_cell in zip(solution_row, sudoku_row) :
                    if not (1 <= solution_cell <= N * M) :
                        return self.rewards["invalid_solution"]
                    if sudoku_cell != 0 and solution_cell != sudoku_cell :
                        return self.rewards["invalid_solution"]
            
            for row in solution :
                if len(set(row)) != N * M :
                    return self.rewards["wrong_solution"]
            for column in range(N * M) :
                if len(set(solution[row][column] for row in range(N * M))) != N * M :
                    return self.rewards["wrong_solution"]
            for i in range(M) :
                for j in range(N) :
                    subgrid = [solution[x][y] for x in range(i * N, (i + 1) * N) for y in range(j * M, (j + 1) * M)]
                    if len(set(subgrid)) != N * M :
                        return self.rewards["wrong_solution"]
            
            return self.rewards["correct_solution"]
        else :
            return self.rewards["wrong_format"]