import random
import numpy as np
from typing import Optional, List
from Gym.environment import VerifiableEnvironment

def magic_square(n):
    if n == 1:
        return np.array([[1]], dtype=int)

    if n % 2 == 1:
        return _magic_odd(n)
    elif n % 4 == 0:
        return _magic_doubly_even(n)
    else:
        raise NotImplementedError("Magic square for singly even n (e.g., 6, 10) is not implemented.")


def _magic_odd(n):
    magic = np.zeros((n, n), dtype=int)
    num = 1
    i, j = 0, n // 2
    while num <= n * n:
        magic[i, j] = num
        num += 1
        ni, nj = (i - 1) % n, (j + 1) % n
        if magic[ni, nj] != 0:
            i = (i + 1) % n
        else:
            i, j = ni, nj
    return magic


def _magic_doubly_even(n):
    magic = np.arange(1, n * n + 1, dtype=int).reshape(n, n)
    for i in range(n):
        for j in range(n):
            if (i % 4 == j % 4) or ((i % 4) + (j % 4) == 3):
                magic[i, j] = n * n + 1 - magic[i, j]
    return magic


def rotate(square):
    return np.rot90(square, random.randint(1, 3))


def mirror(square):
    return np.fliplr(square)


def swap_rows(square, i, j):
    n = square.shape[0]
    A = square.copy()
    A[[i, j], :] = A[[j, i], :]
    c1, c2 = n-1-i, n-1-j
    A[:, [c1, c2]] = A[:, [c2, c1]]
    return square


class MagicSquarePuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Given a grid of size {N} × {N} filled with integers, some cells may be empty (represented by `0`). Please complete the grid to form a **magic square**, such that:
1. Each integer from `1` to `{N}^2` appears **exactly once**.
2. The sum of each row, each column, and both main diagonals is equal to {N} * ({N}^2 + 1) / 2 = {magic_constant}.

The grid is given as follows:
{grid}

**Output Format:** Your final answer should contain {N} lines, each with {N} numbers, separated by spaces. The numbers should represent the completed magic square in **row-major order**, matching the format of the given input."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the MagicSquarePuzzle_Environment instance.
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

        grid = magic_square(N)
        operation_distribution = [0.1, 0.1, 0.8]
        for step in range(N * N) :
            operation = random.choices(["rotate", "mirror", "swap_rows"], weights = operation_distribution)[0]
            if operation == "rotate" :
                grid = rotate(grid)
            elif operation == "mirror" :
                grid = mirror(grid)
            elif operation == "swap_rows" :
                while True :
                    row1, row2 = random.sample(range(N), 2)
                    if row1 != row2 :
                        break
                grid = swap_rows(grid, row1, row2)
            else :
                assert False
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in grid)

        self.parameter["grid"] = grid = [[cell.item() for cell in row] for row in grid]
        assert "sparsity" in self.parameter, "sparsity is required in parameter"
        sparsity = self.parameter["sparsity"]
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        empty_cells = random.sample(range(N * N), max(1, int(N * N * sparsity)))
        for cell in empty_cells :
            row, column = divmod(cell, N)
            grid[row][column] = 0
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            magic_constant = N * (N * N + 1) // 2,
            grid = "\n".join(" ".join(map(str, row)) for row in self.parameter["grid"]),
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
            
            if len(solution) != N or any(len(row) != N for row in solution) :
                return self.rewards["wrong_format"]
            
            if set(cell for row in solution for cell in row) != set(range(1, N * N + 1)) :
                return self.rewards["invalid_solution"]
            if any(original_cell != 0 and cell != original_cell for row, original_row in zip(solution, self.parameter["grid"]) for cell, original_cell in zip(row, original_row)) :
                return self.rewards["invalid_solution"]
            
            satisfied = sum(int(sum(row) == N * (N * N + 1) // 2) for row in solution) + \
                        sum(int(sum(solution[i][j] for i in range(N)) == N * (N * N + 1) // 2) for j in range(N)) + \
                        int(sum(solution[i][i] for i in range(N)) == N * (N * N + 1) // 2) + \
                        int(sum(solution[i][N - i - 1] for i in range(N)) == N * (N * N + 1) // 2)
            assert satisfied <= 2 * N + 2, "satisfied should be less than or equal to 2 * N + 2"

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (2 * N + 2)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (satisfied == (2 * N + 2))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]