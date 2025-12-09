import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class GridBFS_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} grid. Each cell contains `0`, `1`, or `X`. For each cell, compute its **shortest distance** to any cell containing `1`, where distance is defined as the minimum number of steps required to move from one cell to another under the following rules:
1. You may move **up**, **down**, **left**, or **right** to an adjacent cell.
2. You **cannot** move through cells containing `X`.
3. If a cell **cannot reach** any `1`, its distance should be -1.
4. Obviously, the distance for a `1` cell is 0; the distance for an `X` cell is also -1.

The grid is given as follows:
{grid}

**Output Format:** Output {N} lines, each containing {M} integers (separated by spaces), representing the distance of each cell to the nearest `1` cell."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the GridBFS_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)

        cell_distribution = [random.randint(1, N * M) for _ in range(3)]
        cell_distribution = [x / sum(cell_distribution) for x in cell_distribution]
        grid = self.parameter["grid"] = [[random.choices(["0", "1", "X"], weights = cell_distribution)[0] for _ in range(M)] for _ in range(N)]

        distances = self.parameter["gold_answer"] = [[-1] * M for _ in range(N)]
        queue = deque()
        for i in range(N) :
            for j in range(M) :
                if grid[i][j] == "1" :
                    distances[i][j] = 0
                    queue.append((i, j))
        while queue :
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] :
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < M and grid[nx][ny] != "X" and distances[nx][ny] == -1 :
                    distances[nx][ny] = distances[x][y] + 1
                    queue.append((nx, ny))
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in distances)
    

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

            distance = processed_result
            if len(distance) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if not all(len(row) == self.parameter["M"] for row in distance) :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(sum(answer == gold for answer, gold in zip(answer_row, gold_row)) for answer_row, gold_row in zip(distance, self.parameter["gold_answer"])) / (self.parameter["N"] * self.parameter["M"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return distance == self.parameter["gold_answer"]
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]