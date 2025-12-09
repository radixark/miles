import random
from queue import Queue
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Maze_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N}×{N} grid representing a maze. Each cell in the grid is either a wall (`#`) or an open space (`.`). The maze is provided in the following format:
{maze}

Your task is to find the **shortest path** from the top-left corner `(0, 0)` to the bottom-right corner `({N_minus_1}, {N_minus_1})`.
You may move only in the four cardinal directions: **up, down, left, and right**, and only through open spaces (`.`).

**Output Format:**
Your final answer should be a single line containing the sequence of moves, where each move is represented by a character:
- `L` = left
- `R` = right
- `U` = up
- `D` = down
For example, `RRDDLLUU` (do **NOT** include the backticks or quotes) means: right, right, down, down, left, left, up, up.
"""
    action2delta = {
        "L" : (0, -1),
        "R" : (0, +1),
        "U" : (-1, 0),
        "D" : (+1, 0),
    }

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the Maze_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "density" in self.parameter, "density is required in parameter"
        density = self.parameter["density"]
        assert 0.0 <= density < 1.0, "density should be between 0.0 and 1.0"

        while True :
            maze = [["#" if random.random() < density else "." for col in range(N)] for row in range(N)]
            maze[0][0] = maze[N - 1][N - 1] = "."

            prev = [[None] * N for row in range(N)]
            prev[0][0] = (0, 0)
            q = Queue()
            q.put((0, 0))
            while not q.empty() :
                x, y = q.get()
                for (dx, dy) in self.action2delta.values() :
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and maze[nx][ny] == "." and prev[nx][ny] is None :
                        prev[nx][ny] = (x, y)
                        q.put((nx, ny))
            
            if prev[N - 1][N - 1] is not None :
                break
        
        self.parameter["maze"] = ["".join(row) for row in maze]
        
        if prev[N - 1][N - 1] is not None :
            path = []
            x, y = N - 1, N - 1
            while (x, y) != (0, 0) :
                px, py = prev[x][y]
                for action, (dx, dy) in self.action2delta.items() :
                    if (x, y) == (px + dx, py + dy) :
                        path.append(action)
                        break
                x, y = px, py
            path.reverse()
            self.parameter["reference_answer"] = "".join(path)
    
    def _prompt_generate(self) -> str :
        """
        Generate the prompt for the problem.
        """
        N = self.parameter["N"]
        N_minus_1 = N - 1
        return self.prompt_template.format(N = N, N_minus_1 = N_minus_1, maze = "\n".join(self.parameter["maze"]))


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            actions = processed_result
            x, y = 0, 0
            for action in actions :
                if action not in self.action2delta :
                    return self.rewards["wrong_format"]
                dx, dy = self.action2delta[action]

                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.parameter["N"] and 0 <= ny < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if self.parameter["maze"][nx][ny] == "#" :
                    return self.rewards["invalid_solution"]
                x, y = nx, ny
            if (x, y) != (self.parameter["N"] - 1, self.parameter["N"] - 1) :
                return self.rewards["unsuccessful_solution"]
            assert len(actions) >= len(self.parameter["reference_answer"]), "actions should be greater than or equal to reference_answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((len(self.parameter["reference_answer"]) / len(actions)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (len(self.parameter["reference_answer"]) == len(actions))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]