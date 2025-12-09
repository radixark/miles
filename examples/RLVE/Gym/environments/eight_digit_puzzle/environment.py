import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class EightDigitPuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} grid, where each cell contains a digit from `0` to `{NM_minus_1}`. At any time, you can **swap the `0`** with one of its four (existing) neighbors:
- `U` = up
- `D` = down
- `L` = left
- `R` = right

You start with the following grid:
{start_grid}

Your goal is to reach the following grid:
{destination_grid}

**Output Format:** Output a single line containing the sequence of moves made by the `0`, represented by a string of characters (`U`, `D`, `L`, `R`). For example, `RRDDLLUU` (do **NOT** include backticks or quotes) means: right, right, down, down, left, left, up, up."""

    action2delta = {
        "L" : (0, -1),
        "R" : (0, +1),
        "U" : (-1, 0),
        "D" : (+1, 0),
    }

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the EightDigitPuzzle_Environment instance.
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

        start_permutation = list(range(N * M))
        random.shuffle(start_permutation)
        start_grid = self.parameter["start_grid"] = [[start_permutation[i * M + j] for j in range(M)] for i in range(N)]

        assert "steps" in self.parameter, "steps is required in parameter"
        steps = self.parameter["steps"]
        assert steps >= 1, "steps should be greater than or equal to 1"

        self.parameter["zero_i"], self.parameter["zero_j"] = zero_i, zero_j = [(i, j) for i in range(N) for j in range(M) if start_grid[i][j] == 0][0]
        destination_grid = self.parameter["destination_grid"] = [row.copy() for row in start_grid]

        action_distribution = [random.randint(1, N * M) for _ in range(4)]
        action_distribution = [weight / sum(action_distribution) for weight in action_distribution]

        self.parameter["reference_answer"] = ""
        for step in range(steps) :
            while True :
                action = random.choices(["U", "D", "L", "R"], weights = action_distribution, k = 1)[0]
                new_zero_i, new_zero_j = zero_i + self.action2delta[action][0], zero_j + self.action2delta[action][1]
                if 0 <= new_zero_i < N and 0 <= new_zero_j < M :
                    self.parameter["reference_answer"] += action
                    destination_grid[zero_i][zero_j], destination_grid[new_zero_i][new_zero_j] = destination_grid[new_zero_i][new_zero_j], destination_grid[zero_i][zero_j]
                    zero_i, zero_j = new_zero_i, new_zero_j
                    break
        
    
    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            NM_minus_1 = N * M - 1,
            start_grid = "\n".join(" ".join(map(str, row)) for row in self.parameter["start_grid"]),
            destination_grid = "\n".join(" ".join(map(str, row)) for row in self.parameter["destination_grid"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            destination_grid = [row.copy() for row in self.parameter["start_grid"]]
            zero_i, zero_j = self.parameter["zero_i"], self.parameter["zero_j"]

            for action in  processed_result :
                if action not in self.action2delta :
                    return self.rewards["wrong_format"]
                new_zero_i, new_zero_j = zero_i + self.action2delta[action][0], zero_j + self.action2delta[action][1]
                if 0 <= new_zero_i < self.parameter["N"] and 0 <= new_zero_j < self.parameter["M"] :
                    destination_grid[zero_i][zero_j], destination_grid[new_zero_i][new_zero_j] = destination_grid[new_zero_i][new_zero_j], destination_grid[zero_i][zero_j]
                    zero_i, zero_j = new_zero_i, new_zero_j
                else :
                    return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(sum(int(a == b) for a, b in zip(gold_row, answer_row)) for gold_row, answer_row in zip(self.parameter["destination_grid"], destination_grid)) / (self.parameter["N"] * self.parameter["M"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * all(all(a == b for a, b in zip(gold_row, answer_row)) for gold_row, answer_row in zip(self.parameter["destination_grid"], destination_grid))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]