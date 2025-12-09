import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class LightUpPuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} grid. Each cell contains either a number from `0` to `4`, or a character `B` or `W`.
- All `W` cells are considered **white cells** (including those that may be replaced with `L` later).
- All other cells (`0`–`4` or `B`) are considered **black cells**.

You may replace some `W` cells with `L`, indicating the placement of a **light bulb**. A light bulb illuminates its own cell and extends light in all **four directions** (up, down, left, right), stopping when it hits a black cell or the edge of the grid. Please place light bulbs such that:
1. **Each white cell** is illuminated by **at least one** light bulb.
2. No light bulb is illuminated by another light bulb, i.e., no two light bulbs can be placed in the same row or column without a black cell in between.
3. **Each black cell** with a number from `0` to `4` must have **exactly that many** light bulbs in its 4 neighboring cells (up, down, left, right).

The grid is given in **row-major order**:
{grid}

**Output Format:** Output {N} lines, each containing {M} characters with no separators. Some `W` cells should be replaced with `L` to indicate light bulbs; all other cells remain unchanged."""
    
    def __init__(self,
                 black_cell_density_range : tuple = (0.6, 0.95),
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the LightUpPuzzle_Environment instance.
        """
        super().__init__(**kwargs)

        self.black_cell_density_range = black_cell_density_range
        assert len(black_cell_density_range) == 2 and 0.0 < black_cell_density_range[0] < black_cell_density_range[1] < 1.0, "black_cell_density_range should be a tuple of two floats in (0, 1)"

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

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)
        grid = [["W"] * M for _ in range(N)]

        black_cell_density = random.uniform(self.black_cell_density_range[0], self.black_cell_density_range[1])
        black_cells = random.sample(range(N * M), max(1, min(int(N * M * black_cell_density), N * M - 1)))
        for cell in black_cells :
            row, column = divmod(cell, M)
            grid[row][column] = "B"
        
        white_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "W"]
        assert len(white_cells) >= 1, "There should be at least one white cell"
        random.shuffle(white_cells)
        illuminated = [[False] * M for _ in range(N)]
        for i, j in white_cells :
            if illuminated[i][j] :
                continue
            grid[i][j] = "L"
            illuminated[i][j] = True

            for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)) :
                ni, nj = i + di, j + dj
                while 0 <= ni < N and 0 <= nj < M :
                    if grid[ni][nj] == "B" :
                        break
                    assert grid[ni][nj] != "L", "There should be no light bulb in the same row or column without a black cell in between"
                    illuminated[ni][nj] = True
                    ni += di
                    nj += dj
        
        assert "density" in self.parameter, "density is required in parameter"
        density = self.parameter["density"]
        assert 0 < density < 1, "density should be between 0 and 1"
        black_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "B"]
        black_cells = random.sample(black_cells, max(1, int(len(black_cells) * density)))
        assert len(black_cells) > 0, "There should be at least one black cell with a number"
        for i, j in black_cells :
            counting = 0
            for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)) :
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < M and grid[ni][nj] == "L" :
                    counting += 1
            grid[i][j] = str(counting)
        
        self.parameter["reference_answer"] = "\n".join("".join(row) for row in grid)

        self.parameter["grid"] = ["".join(cell if cell != "L" else "W" for cell in row) for row in grid]
    

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

            N, M = self.parameter["N"], self.parameter["M"]
            solution = processed_result
            
            if len(solution) != N or any(len(row) != M for row in solution) :
                return self.rewards["wrong_format"]
            
            for solution_row, original_row in zip(solution, self.parameter["grid"]) :
                for solution_cell, original_cell in zip(solution_row, original_row) :
                    if original_cell == "W" :
                        if solution_cell not in "WL" :
                            return self.rewards["invalid_solution"]
                    elif original_cell in "B01234" :
                        if solution_cell != original_cell :
                            return self.rewards["invalid_solution"]
                    else :
                        assert False, "Unknown cell type: {}".format(original_cell)
            
            illuminated = [[False] * M for _ in range(N)]
            for i in range(N) :
                for j in range(M) :
                    if solution[i][j] == "L" :
                        illuminated[i][j] = True
                        for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)) :
                            ni, nj = i + di, j + dj
                            while 0 <= ni < N and 0 <= nj < M :
                                if solution[ni][nj] != "W" :
                                    if solution[ni][nj] == "L" :
                                        return self.rewards["invalid_solution"]
                                    elif solution[ni][nj] in "B01234" :
                                        break
                                    else :
                                        assert False, "Unknown cell type: {}".format(solution[ni][nj])
                                illuminated[ni][nj] = True
                                ni += di
                                nj += dj
            if any(not illuminated[i][j] for i in range(N) for j in range(M) if self.parameter["grid"][i][j] == "W") :
                return self.rewards["invalid_solution"]
            
            satisfied, total = 0, 0
            for i in range(N) :
                for j in range(M) :
                    if self.parameter["grid"][i][j] in "01234" :
                        total += 1
                        counting = 0
                        for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)) :
                            ni, nj = i + di, j + dj
                            if 0 <= ni < N and 0 <= nj < M and solution[ni][nj] == "L" :
                                counting += 1
                        if counting == int(self.parameter["grid"][i][j]) :
                            satisfied += 1
            assert satisfied <= total and total > 0, "satisfied should be less than or equal to total and total should be greater than 0"

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / total) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == total)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]