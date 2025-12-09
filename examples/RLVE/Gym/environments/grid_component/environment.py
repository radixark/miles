import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class GridComponent_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} grid. Each cell contains either `0` or `1`. Please compute the **largest connected component** of `1`s in the grid, where a connected component is defined as a group of `1` cells that are reachable from each other by moving **up**, **down**, **left**, or **right** to an adjacent `1` cell.

The grid is given as follows:
{grid}

**Output Format:** Output a single integer — the size of the largest connected component (i.e., the number of `1`s in it). If there are no `1`s in the grid, output `0`.
"""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the GridComponent_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)

        one_probability = random.uniform(0.1, 0.9)
        grid = self.parameter["grid"] = ["".join("01"[random.random() < one_probability] for _ in range(M)) for _ in range(N)]

        labels = [[0] * M for _ in range(N)]
        def DFS(x, y) :
            stack = [(x, y)]
            while stack :
                x, y = stack.pop()
                for dx, dy in [(-1, 0), (+1, 0), (0, -1), (0, +1)] :
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < M and grid[nx][ny] == "1" :
                        if labels[nx][ny] == 0 :
                            labels[nx][ny] = labels[x][y]
                            stack.append((nx, ny))
                        else :
                            assert labels[nx][ny] == labels[x][y], "Labels should match for connected components"
        total = 0
        counting = [0]
        for x in range(N) :
            for y in range(M) :
                if grid[x][y] == "1" :
                    if labels[x][y] == 0 :
                        total += 1
                        counting.append(0)
                        labels[x][y] = total
                        DFS(x, y)
                    counting[labels[x][y]] += 1
        self.parameter["reference_answer"] = max(counting)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            grid = "\n".join(self.parameter["grid"]),
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