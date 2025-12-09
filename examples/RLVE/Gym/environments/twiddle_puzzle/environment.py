import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TwiddlePuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} grid, where each cell contains a digit from `0` to `{NM_minus_1}`. At any time, you may select a cell `(i, j)` such that 0 ≤ i ≤ {N} - {K} and 0 ≤ j ≤ {M} - {K}. Then, you perform a **90-degree counterclockwise rotation** on the {K} × {K} subgrid starting at position `(i, j)`.

You start with the following grid:
{start_grid}

Your goal is to transform it into the following grid:
{destination_grid}

**Output Format:** Each action should be written on its own line as `i j`, where `i` and `j` are the row and column indices of the top-left corner of the rotated subgrid. Example: `0 1` (do **NOT** include backticks or quotes). Output one action per line in the order they should be performed."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the TwiddlePuzzle_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)
        K = self.parameter["K"] = random.randint(2, min(N, M))

        start_permutation = list(range(N * M))
        random.shuffle(start_permutation)
        start_grid = self.parameter["start_grid"] = [[start_permutation[i * M + j] for j in range(M)] for i in range(N)]

        assert "steps" in self.parameter, "steps is required in parameter"
        steps = self.parameter["steps"]
        assert steps >= 1, "steps should be greater than or equal to 1"

        destination_grid = [row.copy() for row in start_grid]
        self.parameter["reference_answer"] = ""
        for step in range(steps) :
            i = random.randint(0, N - K)
            j = random.randint(0, M - K)
            self.parameter["reference_answer"] += "{} {}\n".format(i, j)

            new_grid = [row.copy() for row in destination_grid]
            for x in range(K) :
                for y in range(K) :
                    new_grid[i + K - 1 - y][j + x] = destination_grid[i + x][j + y]
            destination_grid = new_grid
        self.parameter["destination_grid"] = destination_grid
    

    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            NM_minus_1 = N * M - 1,
            K = self.parameter["K"],
            start_grid = "\n".join(" ".join(map(str, row)) for row in self.parameter["start_grid"]),
            destination_grid = "\n".join(" ".join(map(str, row)) for row in self.parameter["destination_grid"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            actions = []
            for line in answer.splitlines() :
                line = line.strip()
                if line :
                    actions.append(line.split())
                    action = actions[-1]
                    if len(action) != 2 :
                        return None
                    try :
                        action[0] = int(action[0])
                        action[1] = int(action[1])
                    except ValueError :
                        return None
            return actions
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            destination_grid = [row.copy() for row in self.parameter["start_grid"]]
            
            for i, j in processed_result :
                if not (0 <= i <= self.parameter["N"] - self.parameter["K"] and 0 <= j <= self.parameter["M"] - self.parameter["K"]) :
                    return self.rewards["invalid_solution"]
                new_grid = [destination_grid[row].copy() for row in range(self.parameter["N"])]
                for x in range(self.parameter["K"]) :
                    for y in range(self.parameter["K"]) :
                        new_grid[i + self.parameter["K"] - 1 - y][j + x] = destination_grid[i + x][j + y]
                destination_grid = new_grid
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(sum(int(a == b) for a, b in zip(gold_row, answer_row)) for gold_row, answer_row in zip(self.parameter["destination_grid"], destination_grid)) / (self.parameter["N"] * self.parameter["M"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * all(all(a == b for a, b in zip(gold_row, answer_row)) for gold_row, answer_row in zip(self.parameter["destination_grid"], destination_grid))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]