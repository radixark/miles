import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class WhackAMole_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2484
    prompt_template = \
r"""You are given an {N} × {M} grid, where each cell contains a non-negative integer representing the number of moles in that hole:
{grid}

You are allowed to define a **fixed** hammer size of r × c (1 ≤ r ≤ {N}, 1 ≤ c ≤ {M}) before starting. Each time you swing the hammer:
- You choose an r × c subrectangle in the grid (without rotation).
- This subrectangle must be fully within the grid.
- Each cell in the subrectangle must contain at least 1 mole.
- Each cell in the subrectangle has exactly 1 mole removed (so r × c moles are removed per swing).

You may swing the hammer multiple times, but you cannot change its size after choosing r and c. Your goal is to remove all the moles from the grid with the **minimum number of swings**.

**Output Format:** Your final answer should be a single integer — the **minimum number of hammer swings** required to remove all moles from the grid.
"""

    def __init__(self,
                 max_beat : int = 3,
                 wrong_format : float = -1.0, wrong_answer : float = 0.0, correct_answer : float = +1.0,
                 **kwargs) :
        """
        Initialize the WhackAMole_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_beat = max_beat
        assert max_beat >= 1, "max_beat should be greater than or equal to 1"

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_answer" : wrong_answer,
            "correct_answer" : correct_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"
 
        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)

        R, C = random.randint(1, N), random.randint(1, M)
        grid = self.parameter["grid"] = [[0] * M for _ in range(N)]
        for i in range(N - R + 1) :
            for j in range(M - C + 1) :
                num_moles = random.randint(0, self.max_beat)
                grid[i][j] += num_moles
                if i + R < N :
                    grid[i + R][j] -= num_moles
                if j + C < M :
                    grid[i][j + C] -= num_moles
                if i + R < N and j + C < M :
                    grid[i + R][j + C] += num_moles
        for i in range(N) :
            for j in range(M) :
                if i > 0 :
                    grid[i][j] += grid[i - 1][j]
                if j > 0 :
                    grid[i][j] += grid[i][j - 1]
                if i > 0 and j > 0 :
                    grid[i][j] -= grid[i - 1][j - 1]


        total = sum(sum(row) for row in grid)
        if total == 0 :
            self.parameter["reference_answer"] = 0
            return

        best_area = 0

        # Try every possible hammer size r x c, largest area first
        for area in range(N * M + 1, 0, -1) :
            if total % area != 0:
                continue
            if area <= best_area:
                continue
            for r in range(1, area + 1):
                if area % r != 0:
                    continue
                c = area // r
                if not (1 <= r <= N and 1 <= c <= M):
                    continue
                # Skip if we already have a better or equal area
                if area <= best_area:
                    continue

                # 2D difference array, size (N+1)x(M+1)
                diff = [[0] * (M + 1) for _ in range(N + 1)]
                ok = True

                # Sweep through the grid, maintaining prefix‐sum of diff
                for i in range(N):
                    for j in range(M):
                        # accumulate 2D prefix sum at (i,j)
                        if i > 0:
                            diff[i][j] += diff[i - 1][j]
                        if j > 0:
                            diff[i][j] += diff[i][j - 1]
                        if i > 0 and j > 0:
                            diff[i][j] -= diff[i - 1][j - 1]

                        # If we've hit more moles here than exist, fail
                        if diff[i][j] > grid[i][j]:
                            ok = False
                            break

                        # If we haven't hit enough, schedule hammer swings
                        if diff[i][j] < grid[i][j]:
                            # Must be able to place an r×c rectangle here
                            if i + r > N or j + c > M:
                                ok = False
                                break
                            t = grid[i][j] - diff[i][j]
                            # 2D-difference updates for adding t to rectangle [i..i+r-1][j..j+c-1]
                            diff[i][j]         += t
                            diff[i + r][j]     -= t
                            diff[i][j + c]     -= t
                            diff[i + r][j + c] += t
                    if not ok:
                        break

                if ok:
                    best_area = area

        # The minimum number of swings is total moles divided by the largest valid hammer area
        assert best_area >= R * C, "best_area should be at least R * C"
        self.parameter["reference_answer"] = total // best_area
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            grid = "\n".join(" ".join(map(str, row)) for row in self.parameter["grid"]),
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]