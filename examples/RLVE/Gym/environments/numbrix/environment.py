import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Numbrix_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} matrix with some cells filled with numbers from `0` to `{NM_minus_1}`, and some cells empty (represented by `-1`). Please fill the empty cells with numbers from `0` to `{NM_minus_1}` such that:
1. Each number from `0` to `{NM_minus_1}` appears **exactly once** in the matrix.
2. Each number is **horizontally or vertically adjacent** to the next number (i.e., every number `x` is adjacent to `x + 1`).

The matrix is given as follows:
{matrix}

**Output Format:** Your final answer should contain {N} lines, each with {M} numbers, separated by spaces. The numbers should represent the completed matrix in **row-major order**, matching the format of the given input."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(1/path)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the Numbrix_Environment instance.
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

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def is_inside(x, y):
            return 0 <= x < N and 0 <= y < M

        def count_unvisited_degree(x, y):
            cnt = 0
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if is_inside(nx, ny) and not visited[nx][ny]:
                    cnt += 1
            return cnt

        def check_connectivity(remain):
            start = None
            for i in range(N):
                for j in range(M):
                    if not visited[i][j]:
                        start = (i, j)
                        break
                if start:
                    break
            if not start:
                return True
            stack = [start]
            seen = {start}
            count = 1
            while stack:
                x, y = stack.pop()
                for dx, dy in dirs:
                    xx, yy = x + dx, y + dy
                    if is_inside(xx, yy) and not visited[xx][yy] and (xx, yy) not in seen:
                        seen.add((xx, yy))
                        stack.append((xx, yy))
                        count += 1
            return count == remain

        def DFS(step, x, y):
            if step == N * M:
                return True
            cand = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if is_inside(nx, ny) and not visited[nx][ny]:
                    cand.append((nx, ny))
            if not cand:
                return False
            random.shuffle(cand)
            cand_scores = []
            for nx, ny in cand:
                deg = count_unvisited_degree(nx, ny)
                cand_scores.append((deg, nx, ny))
            cand_scores.sort(key=lambda t: t[0])
            for _, nx, ny in cand_scores:
                visited[nx][ny] = True
                order[nx][ny] = step
                path.append((nx, ny))
                remain = N * M - (step + 1)
                if check_connectivity(remain):
                    if DFS(step + 1, nx, ny):
                        return True
                visited[nx][ny] = False
                order[nx][ny] = -1
                path.pop()
            return False

        def generate_random_hamiltonian_path():
            global visited, order, path
            while True:
                sx = random.randint(0, N - 1)
                sy = random.randint(0, M - 1)
                visited = [[False] * M for _ in range(N)]
                order = [[-1] * M for _ in range(N)]
                path = []
                visited[sx][sy] = True
                order[sx][sy] = 0
                path = [(sx, sy)]
                if DFS(1, sx, sy):
                    return path, order
        
        self.parameter["matrix"] = matrix = generate_random_hamiltonian_path()[-1]
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in matrix)

        assert "sparsity" in self.parameter, "sparsity is required in parameter"
        sparsity = self.parameter["sparsity"]
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        empty_cells = random.sample(range(N * M), max(1, int(N * M * sparsity)))
        for cell in empty_cells :
            row, column = divmod(cell, M)
            matrix[row][column] = -1
    

    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            NM_minus_1 = N * M - 1,
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

            N, M = self.parameter["N"], self.parameter["M"]
            solution = processed_result
            
            if len(solution) != N or any(len(row) != M for row in solution) :
                return self.rewards["wrong_format"]
            
            location = [None] * (N * M)
            i = 0
            for original_row, solution_row in zip(self.parameter["matrix"], solution) :
                j = 0
                for original_value, solution_value in zip(original_row, solution_row) :
                    if original_value != -1 and original_value != solution_value :
                        return self.rewards["invalid_solution"]
                    if not (0 <= solution_value < N * M) :
                        return self.rewards["invalid_solution"]
                    if location[solution_value] is not None :
                        return self.rewards["invalid_solution"]
                    location[solution_value] = (i, j)
                    j += 1
                i += 1
            
            path = 1
            for value in range(N * M - 1) :
                assert location[value] is not None, "location[{}] should not be None".format(value)
                assert location[value + 1] is not None, "location[{}] should not be None".format(value + 1)
                x1, y1 = location[value]
                x2, y2 = location[value + 1]
                path += int(abs(x1 - x2) + abs(y1 - y2) != 1)

            if self.rewards["rewarding_strategy"] == "(1/path)^beta" :
                return self.rewards["rewarding_weight"] * ((1 / path) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "path=1" :
                return self.rewards["rewarding_weight"] * (path == 1)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]