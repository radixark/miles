import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class NinePuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} grid, where each cell contains a digit from `0` to `{NM_minus_1}`.

At any time, you may perform one of the following actions:
- Pick a row i (0 ≤ i < {N}) and shift it left or right by **at most** {row_K} cells.
- Pick a column j (0 ≤ j < {M}) and shift it up or down by **at most** {col_K} cells.

You start with the following grid:
{start_grid}

Your goal is to transform it into the following grid:
{destination_grid}

**Output Format:** Each action should be written on its own line in the following format: `[row_or_column] [index] [shifts]`
Where:
- `row_or_column` is either `row` or `column`
- `index` is the 0-based index of the row or column
- `shifts` is a signed integer: positive for right/down, negative for left/up
- Example: `row 0 2` or `column 1 -3`
Do **NOT** include backticks or quotes in your output. Output one action per line in the order they should be performed."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the NinePuzzle_Environment instance.
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
        row_K, col_K = self.parameter["row_K"], self.parameter["col_K"] = random.randint(1, M - 1), random.randint(1, N - 1)

        start_permutation = list(range(N * M))
        random.shuffle(start_permutation)
        start_grid = self.parameter["start_grid"] = [[start_permutation[i * M + j] for j in range(M)] for i in range(N)]

        assert "steps" in self.parameter, "steps is required in parameter"
        steps = self.parameter["steps"]
        assert steps >= 1, "steps should be greater than or equal to 1"

        destination_grid = [row.copy() for row in start_grid]
        self.parameter["reference_answer"] = ""
        for step in range(steps) :
            row_or_column = random.choice(["row", "column"])
            index = random.randint(0, N - 1) if row_or_column == "row" else random.randint(0, M - 1)
            while True :
                shifts = random.randint(-row_K, row_K) if row_or_column == "row" else random.randint(-col_K, col_K)
                if shifts != 0 :
                    break
            self.parameter["reference_answer"] += "{} {} {}\n".format(row_or_column, index, shifts)

            new_grid = [row.copy() for row in destination_grid]
            if row_or_column == "row" :
                assert abs(shifts) <= M - 1
                assert abs(shifts) <= row_K
                for j in range(M) :
                    new_grid[index][j] = destination_grid[index][((j - shifts) % M + M) % M]
            else :
                assert row_or_column == "column"
                assert abs(shifts) <= N - 1
                assert abs(shifts) <= col_K
                for i in range(N) :
                    new_grid[i][index] = destination_grid[((i - shifts) % N + N) % N][index]
            destination_grid = new_grid
        self.parameter["destination_grid"] = destination_grid
    

    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            NM_minus_1 = N * M - 1,
            row_K = self.parameter["row_K"],
            col_K = self.parameter["col_K"],
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
                    if len(action) != 3 :
                        return None
                    if action[0] not in ("row", "column") :
                        return None
                    try :
                        action[1] = int(action[1])
                        action[2] = int(action[2])
                    except ValueError :
                        return None
            return actions
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            destination_grid = [row.copy() for row in self.parameter["start_grid"]]
            
            for action in processed_result :
                new_grid = [row.copy() for row in destination_grid]
                if action[0] == "row" :
                    index = action[1]
                    if not (0 <= index < self.parameter["N"]) :
                        return self.rewards["invalid_solution"]
                    shifts = action[2]
                    if not (-self.parameter["row_K"] <= shifts <= self.parameter["row_K"]) :
                        return self.rewards["invalid_solution"]
                    for j in range(self.parameter["M"]) :
                        new_grid[index][j] = destination_grid[index][((j - shifts) % self.parameter["M"] + self.parameter["M"]) % self.parameter["M"]]
                else :
                    assert action[0] == "column"
                    index = action[1]
                    if not (0 <= index < self.parameter["M"]) :
                        return self.rewards["invalid_solution"]
                    shifts = action[2]
                    if not (-self.parameter["col_K"] <= shifts <= self.parameter["col_K"]) :
                        return self.rewards["invalid_solution"]
                    for i in range(self.parameter["N"]) :
                        new_grid[i][index] = destination_grid[((i - shifts) % self.parameter["N"] + self.parameter["N"]) % self.parameter["N"]][index]
                destination_grid = new_grid
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(sum(int(a == b) for a, b in zip(gold_row, answer_row)) for gold_row, answer_row in zip(self.parameter["destination_grid"], destination_grid)) / (self.parameter["N"] * self.parameter["M"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * all(all(a == b for a, b in zip(gold_row, answer_row)) for gold_row, answer_row in zip(self.parameter["destination_grid"], destination_grid))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]