import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment

class QueenPlacement_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an {N} × {N} chessboard grid. Some cells already contain queens (denoted by 'Q'), and the rest are empty ('.').
{grid}

Please place **{K} additional queens** such that **no two queens threaten each other**. A queen threatens another if they share the same **row**, **column**, or **diagonal** (both main and anti-diagonals).

**Output Format:** Output {N} lines, each containing a string of length {N}. Each string represents a row of the grid using 'Q' for a queen and '.' for an empty cell."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, wrong_solution : float = 0.0, correct_solution : float = 1.0,
                 **kwargs) :
        """
        Initialize the QueenPlacement_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "wrong_solution": wrong_solution,
            "correct_solution": correct_solution,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        grid = self.parameter["grid"] = [["." for _ in range(N)] for _ in range(N)]

        all_cells = [(i, j) for i in range(N) for j in range(N)]
        random.shuffle(all_cells)

        row, col, main_diag, anti_diag = set(), set(), set(), set()
        queens = []
        for i, j in all_cells :
            if i in row or j in col or (i - j) in main_diag or (i + j) in anti_diag :
                continue
            grid[i][j] = "Q"
            queens.append((i, j))
            row.add(i)
            col.add(j)
            main_diag.add(i - j)
            anti_diag.add(i + j)
        self.parameter["reference_answer"] = "\n".join("".join(row) for row in grid)
        
        K = self.parameter["K"] = random.randint(1, max(1, len(queens) // 2))

        queens = random.sample(queens, K)
        for i, j in queens :
            grid[i][j] = "."
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            grid = "\n".join("".join(row) for row in self.parameter["grid"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            grid = []
            for line in answer.splitlines() :
                line = line.strip()
                if line :
                    grid.append(line)
            return grid
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            grid = processed_result
            if len(grid) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if any(len(row) != self.parameter["N"] for row in grid) :
                return self.rewards["wrong_format"]
            if any(cell not in "Q." for row in grid for cell in row) :
                return self.rewards["wrong_format"]
            
            counting = 0
            row, col, main_diag, anti_diag = set(), set(), set(), set()
            i = 0
            for original_row, current_row in zip(self.parameter["grid"], grid) :
                j = 0
                for original_cell, current_cell in zip(original_row, current_row) :
                    if original_cell == "Q" :
                        if current_cell != "Q" :
                            return self.rewards["invalid_solution"]
                    else :
                        assert original_cell == ".", "original cell should be empty"
                        counting += (current_cell == "Q")
                    if current_cell == "Q" :
                        if i in row or j in col or (i - j) in main_diag or (i + j) in anti_diag :
                            return self.rewards["wrong_solution"]
                        row.add(i)
                        col.add(j)
                        main_diag.add(i - j)
                        anti_diag.add(i + j)
                    j += 1
                i += 1
            
            if counting != self.parameter["K"] :
                return self.rewards["wrong_solution"]
            else :
                return self.rewards["correct_solution"]
        else :
            return self.rewards["wrong_format"]