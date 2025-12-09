import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Minesweeping_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} matrix. Each element is either a number in [0, 8] or `-1`. Your task is to construct a grid of the same size, satisfying the following conditions:
1. Each cell is either `*` or `.`
2. For any cell in the original matrix that is **NOT** `-1`, the corresponding cell in the output grid must be `.`. Also, its number must equal the number of `*` characters in its **8 neighboring cells**.

The matrix is given in **row-major order**:
{matrix}

**Output Format:** Output {N} lines, each containing {M} characters with no separators. Each character must be either `*` or `.`"""

    def __init__(self,
                 mine_density_range : tuple = (0.4, 0.7),
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the Minesweeping_Environment instance.
        """
        super().__init__(**kwargs)

        self.mine_density_range = mine_density_range
        assert len(mine_density_range) == 2 and 0.0 < mine_density_range[0] < mine_density_range[1] < 1.0, "mine_density_range should be a tuple of two floats in (0, 1)"

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
        self.parameter["grid"] = grid = [["."] * M for _ in range(N)]

        mine_density = random.uniform(self.mine_density_range[0], self.mine_density_range[1])
        mine_cells = random.sample(range(N * M), max(1, min(int(N * M * mine_density), N * M - 1)))
        for cell in mine_cells :
            row, column = divmod(cell, M)
            grid[row][column] = "*"
        self.parameter["reference_answer"] = "\n".join("".join(row) for row in grid)

        empty_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "."]
        assert len(empty_cells) >= 1, "There should be at least one empty cell"
        assert "density" in self.parameter, "density is required in parameter"
        density = self.parameter["density"]
        assert 0 < density < 1, "density should be between 0 and 1"
        empty_cells = random.sample(empty_cells, max(1, int(len(empty_cells) * density)))
        for i, j in empty_cells :
            counting = 0
            for di in (-1, 0, +1) :
                for dj in (-1, 0, +1) :
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M and grid[ni][nj] == "*" :
                        counting += 1
            grid[i][j] = counting
        
        for i in range(N) :
            for j in range(M) :
                if grid[i][j] in (".", "*") :
                    grid[i][j] = -1
                else :
                    assert 0 <= grid[i][j] <= 8
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            matrix = "\n".join(" ".join(map(str, row)) for row in self.parameter["grid"]),
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
            if not all(all(c in "*." for c in row) for row in solution) :
                return self.rewards["wrong_format"]
            
            satisfied, total = 0, 0
            for i in range(N) :
                for j in range(M) :
                    if self.parameter["grid"][i][j] != -1 :
                        if solution[i][j] != "." :
                            return self.rewards["invalid_solution"]
                        counting = 0
                        for di in (-1, 0, +1) :
                            for dj in (-1, 0, +1) :
                                if di == 0 and dj == 0 :
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < N and 0 <= nj < M and solution[ni][nj] == "*" :
                                    counting += 1
                        assert 0 <= counting <= 8, "counting should be between 0 and 8"
                        total += 1
                        satisfied += int(counting == self.parameter["grid"][i][j])

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / total) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == total)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]