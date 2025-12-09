import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Kakurasu_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} grid (1-indexed). Fill the grid with `0`s and `1`s such that:
- For each row `i`, the sum of the **column indices** where there are `1`s is equal to `A[i]`. Array `A` is given as: {A}
- For each column `j`, the sum of the **row indices** where there are `1`s is equal to `B[j]`. Array `B` is given as: {B}

**Output Format:** Your final answer should consist of {N} lines, each containing {M} characters (`0` or `1`, with no separators), representing the filled grid."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the Kakurasu_Environment instance.
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
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)

        one_rate = random.uniform(0.1, 0.9)
        grid = [["1" if random.random() < one_rate else "0" for _ in range(M)] for _ in range(N)]
        self.parameter["reference_answer"] = "\n".join("".join(row) for row in grid)

        A = self.parameter["A"] = [sum((j + 1) for j in range(M) if grid[i][j] == "1") for i in range(N)]
        B = self.parameter["B"] = [sum((i + 1) for i in range(N) if grid[i][j] == "1") for j in range(M)]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            A = " ".join("A[{}]={}".format(i + 1, a) for i, a in enumerate(self.parameter["A"])),
            B = " ".join("B[{}]={}".format(j + 1, b) for j, b in enumerate(self.parameter["B"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            image = []
            for line in answer.splitlines() :
                line = line.strip()
                if line :
                    image.append(line)
            return image
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            grid = processed_result
            if len(grid) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if not all(len(row) == self.parameter["M"] for row in grid) :
                return self.rewards["wrong_format"]
            if not all(cell in "01" for row in grid for cell in row) :
                return self.rewards["wrong_format"]

            A = [sum((j + 1) for j in range(self.parameter["M"]) if grid[i][j] == "1") for i in range(self.parameter["N"])]
            B = [sum((i + 1) for i in range(self.parameter["N"]) if grid[i][j] == "1") for j in range(self.parameter["M"])]
            assert len(A) == len(self.parameter["A"]) and len(B) == len(self.parameter["B"]), "Length of A or B does not match the expected length"

            satisfied = sum(int(a == gold_a) for a, gold_a in zip(A, self.parameter["A"])) + \
                        sum(int(b == gold_b) for b, gold_b in zip(B, self.parameter["B"]))
            assert satisfied <= (self.parameter["N"] + self.parameter["M"]), "Satisfied count exceeds the number of rows and columns"
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (self.parameter["N"] + self.parameter["M"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (satisfied == (self.parameter["N"] + self.parameter["M"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]