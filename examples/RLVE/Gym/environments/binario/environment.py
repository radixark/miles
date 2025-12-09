import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Binario_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} matrix. Each cell contains either '0', '1', or '*' ('*' means the cell is empty). Please fill all '*' cells with either '0' or '1' such that:
1. The number of `1`s in each row (from top to bottom) is: {row_counts}.
2. The number of `1`s in each column (from left to right) is: {col_counts}.
3. No more than two consecutive cells in a row or column can contain the same number.

The matrix is given in **row-major order**, with each row represented as a string of '0', '1', and '*':
{matrix}

**Output Format:** Output {N} lines, each containing {M} characters, where each character is either '0' or '1'. The output should match the format of the input (i.e., one row per line, no separators)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the Binario_Environment instance.
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

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)

        def generate_matrix(N, M):
            # Initialize the grid with None
            grid = [[None] * M for _ in range(N)]

            all_cells = [(i, j) for i in range(N) for j in range(M)]
            random.shuffle(all_cells)  # Shuffle to ensure randomness in placement

            backtrack_counting = 0

            def backtrack(idx):
                # If we've filled past the last row, we're done
                if idx == len(all_cells):
                    return True
                i, j = all_cells[idx]

                nonlocal backtrack_counting
                backtrack_counting += 1
                if backtrack_counting > 10000000:
                    return False

                # Try placing 0 or 1 in random order
                for v in random.sample(["0", "1"], 2):
                    # Check adjacency constraints in row (no three in a row)
                    if j >= 2 and grid[i][j-1] == v and grid[i][j-2] == v:
                        continue
                    if j >= 1 and j + 1 < M and grid[i][j-1] == v and grid[i][j+1] == v:
                        continue
                    if j + 2 < M and grid[i][j+1] == v and grid[i][j+2] == v:
                        continue

                    # Check adjacency constraints in column
                    if i >= 2 and grid[i-1][j] == v and grid[i-2][j] == v:
                        continue
                    if i >= 1 and i + 1 < N and grid[i-1][j] == v and grid[i+1][j] == v:
                        continue
                    if i + 2 < N and grid[i+1][j] == v and grid[i+2][j] == v:
                        continue

                    # Place v
                    grid[i][j] = v

                    # Recurse
                    if backtrack(idx + 1):
                        return True

                    grid[i][j] = None

                # No valid value at (i, j): backtrack
                return False

            return grid if backtrack(0) else None
        
        matrix = generate_matrix(N, M)
        if matrix is None :
            self.parameter = None
            return
        self.parameter["reference_answer"] = "\n".join("".join(row) for row in matrix)

        self.parameter["row_counts"] = [sum(int(cell == "1") for cell in row) for row in matrix]
        self.parameter["col_counts"] = [sum(int(matrix[i][j] == "1") for i in range(N)) for j in range(M)]

        assert "sparsity" in self.parameter, "sparsity is required in parameter"
        sparsity = self.parameter["sparsity"]
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        empty_cells = random.sample(range(N * M), max(1, int(N * M * sparsity)))
        for cell in empty_cells :
            row, column = divmod(cell, M)
            matrix[row][column] = '*'
        self.parameter["matrix"] = ["".join(row) for row in matrix]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            matrix = "\n".join("".join(map(str, row)) for row in self.parameter["matrix"]),
            row_counts = ", ".join(map(str, self.parameter["row_counts"])),
            col_counts = ", ".join(map(str, self.parameter["col_counts"])),
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
            for row in solution :
                if not all(c in "01" for c in row) :
                    return self.rewards["wrong_format"]
            
            for row, original_row in zip(solution, self.parameter["matrix"]) :
                for cell, original_cell in zip(row, original_row) :
                    if original_cell != '*' and cell != original_cell :
                        assert (original_cell == '0' and cell == '1') or (original_cell == '1' and cell == '0')
                        return self.rewards["invalid_solution"]
            
            delta = [
                (+1, 0),
                (-1, 0),
                (0, +1),
                (0, -1),
            ]
            for i in range(N) :
                for j in range(M) :
                    for di, dj in delta :
                        ni, nj = i + di, j + dj
                        nni, nnj = i + 2 * di, j + 2 * dj
                        if 0 <= ni < N and 0 <= nj < M and 0 <= nni < N and 0 <= nnj < M :
                            if solution[i][j] == solution[ni][nj] == solution[nni][nnj] :
                                return self.rewards["invalid_solution"]
            
            row_counts = [sum(int(cell == "1") for cell in row) for row in solution]
            col_counts = [sum(int(solution[i][j] == "1") for i in range(N)) for j in range(M)]

            satisfied = sum(int(answer == gold) for answer, gold in zip(row_counts, self.parameter["row_counts"])) + \
                        sum(int(answer == gold) for answer, gold in zip(col_counts, self.parameter["col_counts"]))
            assert satisfied <= N + M, "satisfied should not exceed N + M"

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (N + M)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == (N + M))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]