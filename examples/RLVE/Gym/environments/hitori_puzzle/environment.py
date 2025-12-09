import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class HitoriPuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} matrix. Each cell contains an integer. Please "black out" some cells such that:
1. In each row and each column, no number appears more than once **among the remaining (non-blacked-out) cells**.
2. No two blacked-out cells are **adjacent** (horizontally or vertically).
3. All remaining cells must form a **single connected region** — you must be able to reach any remaining cell from any other by moving up, down, left, or right.

The matrix is given in **row-major order**, with each row represented as a list of integers separated by spaces:
{matrix}

**Output Format:** Output {N} lines, each containing {M} characters with no separators (also in **row-major order**). Use `.` for a remaining cell and `*` for a blacked-out cell."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the HitoriPuzzle_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def check_connected(self, grid, N, M) :
        visited = [[False] * M for _ in range(N)]
        def DFS(x, y) :
            visited[x][y] = True
            for dx, dy in [(-1, 0), (+1, 0), (0, -1), (0, +1)] :
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < M and not visited[nx][ny] and grid[nx][ny] == "." :
                    DFS(nx, ny)
        for i in range(N) :
            for j in range(M) :
                if grid[i][j] == "." :
                    DFS(i, j)
                    return all(visited[_i][_j] for _i in range(N) for _j in range(M) if grid[_i][_j] == ".")
        assert False
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)

        def generate(N, M) :
            matrix = [[None] * M for _ in range(N)]
            reference_answer = [["."] * M for _ in range(N)]

            all_cells = [(i, j) for i in range(N) for j in range(M)]
            random.shuffle(all_cells)

            def backtrack(idx) :
                if idx == len(all_cells) :
                    return True
                i, j = all_cells[idx]

                remaining_numbers = set(matrix[i][_j] for _j in range(M) if reference_answer[i][_j] == "." and matrix[i][_j] is not None) | \
                                    set(matrix[_i][j] for _i in range(N) if reference_answer[_i][j] == "." and matrix[_i][j] is not None)
                
                for color in random.sample([".", "*"], 2) :
                    if color == "." :
                        num = 0
                        while num in remaining_numbers :
                            num += 1
                        matrix[i][j] = num
                    else :
                        if not remaining_numbers :
                            continue
                        ok = True
                        for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)] :
                            ni, nj = i + di, j + dj
                            if 0 <= ni < N and 0 <= nj < M and reference_answer[ni][nj] == "*" :
                                ok = False
                                break
                        if not ok :
                            continue
                        reference_answer[i][j] = "*"
                        if not self.check_connected(reference_answer, N, M) :
                            reference_answer[i][j] = "."
                            continue
                        matrix[i][j] = random.choice(list(remaining_numbers))
                    assert backtrack(idx + 1)
                    return True
                
                return False
            
            assert backtrack(0), "Failed to generate a valid matrix"
            return matrix, reference_answer
        
        self.parameter["matrix"], self.parameter["reference_answer"] = generate(N, M)
        self.parameter["reference_answer"] = "\n".join("".join(row) for row in self.parameter["reference_answer"])
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            matrix = "\n".join(" ".join(map(str, row)) for row in self.parameter["matrix"]),
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
            if not all(c in ".*" for row in solution for c in row) :
                return self.rewards["wrong_format"]
            
            for i in range(N) :
                for j in range(M) :
                    if solution[i][j] == "*" :
                        for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)] :
                            ni, nj = i + di, j + dj
                            if 0 <= ni < N and 0 <= nj < M and solution[ni][nj] == "*" :
                                return self.rewards["invalid_solution"]
            if not self.check_connected(solution, N, M) :
                return self.rewards["invalid_solution"]

            satisfied = 0
            for i in range(N) :
                row_numbers = [self.parameter["matrix"][i][j] for j in range(M) if solution[i][j] == "."]
                if len(row_numbers) == len(set(row_numbers)) :
                    satisfied += 1
            for j in range(M) :
                col_numbers = [self.parameter["matrix"][i][j] for i in range(N) if solution[i][j] == "."]
                if len(col_numbers) == len(set(col_numbers)) :
                    satisfied += 1
            assert satisfied <= N + M, "satisfied should not exceed N + M"

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (N + M)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == (N + M))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]