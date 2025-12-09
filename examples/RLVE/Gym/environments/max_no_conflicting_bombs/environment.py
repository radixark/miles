import sys
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaxNoConflictingBombs_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2825
    prompt_template = \
r"""You are given a {N} × {M} grid. Each cell contains one of the following characters: `#`, `x`, or `*`. You may replace some `*` cells with `B`, under the following condition: no two `B` cells may appear in the same row or column **unless** there is at least one `#` between them (i.e., every pair of `B`s in the same row or column must be separated by at least one `#`). Try your best to maximize the number of `B` cells.

The grid is given in **row-major order**:
{grid}

**Output Format:** Output {N} lines, each containing {M} characters with no separators. Replace selected `*` cells with `B`; all other cells should remain unchanged."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the MaxNoConflictingBombs_Environment instance.
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

        distribution = [random.randint(1, N * M) for _ in range(3)]
        distribution = [x / sum(distribution) for x in distribution]
        A = self.parameter["grid"] = [random.choices(["#", "x", "*"], weights = distribution, k = M) for _ in range(N)]


        # Assign row-segment IDs to each non-# cell
        ROW = [[-1] * M for _ in range(N)]
        tot = 0
        for i in range(N):
            j = 0
            while j < M:
                if A[i][j] == '#':
                    j += 1
                else:
                    # start of a new horizontal segment
                    k = j
                    while k < M and A[i][k] != '#':
                        ROW[i][k] = tot
                        k += 1
                    tot += 1
                    j = k
        row_cnt = tot

        # Assign column-segment IDs to each non-# cell
        COL = [[-1] * M for _ in range(N)]
        tot = 0
        for j in range(M):
            i = 0
            while i < N:
                if A[i][j] == '#':
                    i += 1
                else:
                    # start of a new vertical segment
                    k = i
                    while k < N and A[k][j] != '#':
                        COL[k][j] = tot
                        k += 1
                    tot += 1
                    i = k
        col_cnt = tot

        # Build bipartite graph: row segments 0..row_cnt-1 to col segments 0..col_cnt-1
        G = [[] for _ in range(row_cnt)]
        for i in range(N):
            for j in range(M):
                if A[i][j] == '*':
                    u = ROW[i][j]
                    v = COL[i][j]
                    G[u].append(v)

        # Maximum bipartite matching via DFS
        MATCH = [-1] * col_cnt

        # Ensure recursion limit is high enough
        sys.setrecursionlimit(10000)

        def dfs(u, seen):
            for v in G[u]:
                if not seen[v]:
                    seen[v] = True
                    if MATCH[v] == -1 or dfs(MATCH[v], seen):
                        MATCH[v] = u
                        return True
            return False

        result = 0
        for u in range(row_cnt):
            seen = [False] * col_cnt
            if dfs(u, seen):
                result += 1

        self.parameter["gold_answer"] = result
    

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
                    if original_cell == "*" :
                        if solution_cell not in "*B" :
                            return self.rewards["invalid_solution"]
                    else :
                        assert original_cell in "#x", "Original cell should be either '#' or 'x'"
                        if solution_cell != original_cell :
                            return self.rewards["invalid_solution"]
            
            for i in range(N) :
                for j in range(M) :
                    if solution[i][j] == 'B' :
                        for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)) :
                            ni, nj = i + di, j + dj
                            while 0 <= ni < N and 0 <= nj < M :
                                if solution[ni][nj] == 'B' :
                                    return self.rewards["invalid_solution"]
                                if solution[ni][nj] == '#' :
                                    break
                                ni += di
                                nj += dj
            
            answer, gold = sum(row.count('B') for row in solution), self.parameter["gold_answer"]
            assert answer <= gold, "Answer should not exceed the gold answer"

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                if gold == 0 :
                    assert answer == 0, "If gold answer is 0, answer should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((answer / self.parameter["gold_answer"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == self.parameter["gold_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]