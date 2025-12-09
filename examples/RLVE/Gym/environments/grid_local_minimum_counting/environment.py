import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GridLocalMinimumCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3160
    prompt_template = \
r"""Consider a grid of size {N} × {M}, where the numbers from 1 to {N} × {M} are placed in the cells such that **each number appears exactly once**.
A cell is considered a local minimum if its value is strictly less than all of its 8 neighbors (adjacent vertically, horizontally, or diagonally); if a neighbor does not exist, it is considered to be infinitely large. You are given a grid of size {N} × {M} where some cells are marked with `X` and others with `.`. Please count how many valid numberings exist such that the local minima are **exactly** those marked with `X`. The grid is given as follows:
{grid}

**Output Format:** Output a single integer — the number of valid labelings."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the GridLocalMinimumCountingProblem instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)

        permutation = list(range(1, N * M + 1))
        random.shuffle(permutation)
        def get_num(i, j) :
            return permutation[i * M + j]
        self.parameter["grid"] = grid = [['.'] * M for _ in range(N)]
        for i in range(N) :
            for j in range(M) :
                local_minimum = True
                for dx, dy in [(-1, -1), (-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, 0), (+1, +1)] :
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < N and 0 <= nj < M and get_num(ni, nj) <= get_num(i, j) :
                        local_minimum = False
                        break
                if local_minimum :
                    grid[i][j] = 'X'
        

        def compute(raw):
            # Build boolean map of required local minima
            grid = [[(raw[i][j] == 'X') for j in range(M)] for i in range(N)]

            # Quick invalid check: no two required 'X's may be adjacent (including diagonals)
            for i in range(N):
                for j in range(M):
                    if grid[i][j]:
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < N and 0 <= nj < M and grid[ni][nj]:
                                    assert False, "Invalid grid: two local minima are adjacent"
                                    return

            ans = 0

            def inrange(x, y):
                return 0 <= x < N and 0 <= y < M

            def calc():
                """
                For the current grid of local-minima flags, use inclusion-exclusion DP
                to count the number of labelings of the N*M cells so that exactly these
                cells are local minima.
                """
                pos = [(i, j) for i in range(N) for j in range(M) if grid[i][j]]
                cntX = len(pos)
                total = N * M

                # dp[used_cells][subset_mask]
                # We need rows up to total+1 because we transition from i=total -> i+1=total+1
                dp = [[0] * (1 << cntX) for _ in range(total + 2)]
                dp[0][0] = 1

                for s in range(1 << cntX):
                    # mark all cells "blocked" by the minima NOT in subset s
                    blocked = [[False] * M for _ in range(N)]
                    free_cells = total
                    for k in range(cntX):
                        if not (s & (1 << k)):
                            x, y = pos[k]
                            for di in (-1, 0, 1):
                                for dj in (-1, 0, 1):
                                    ni, nj = x + di, y + dj
                                    if inrange(ni, nj) and not blocked[ni][nj]:
                                        blocked[ni][nj] = True
                                        free_cells -= 1

                    for used in range(free_cells + 1):
                        v = dp[used][s]
                        if not v:
                            continue
                        # place a non-min in one of the remaining free cells
                        dp[used + 1][s] += v * (free_cells - used)
                        # or turn one of the excluded minima into an actual minima
                        for k in range(cntX):
                            if not (s & (1 << k)):
                                dp[used + 1][s | (1 << k)] += v

                # We want all total cells assigned, and all minima chosen
                return dp[total][(1 << cntX) - 1]

            def dfs(i, j, sign):
                nonlocal ans
                if i == N:
                    ans += sign * calc()
                    return

                # move to next cell
                ni, nj = (i, j + 1) if j + 1 < M else (i + 1, 0)

                # option 1: don't add a minima here
                dfs(ni, nj, sign)

                # option 2: if this cell is not already a minima, and none of its neighbors is one, we can add it
                if not grid[i][j]:
                    ok = True
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ai, aj = i + di, j + dj
                            if inrange(ai, aj) and grid[ai][aj]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        grid[i][j] = True
                        dfs(ni, nj, -sign)
                        grid[i][j] = False

            dfs(0, 0, 1)
            assert ans > 0
            return ans
        self.parameter["reference_answer"] = compute(grid)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            grid = "\n".join("".join(row) for row in self.parameter["grid"]),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]