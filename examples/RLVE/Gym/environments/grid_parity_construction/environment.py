import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class GridParityConstruction_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Please construct a {N} × {M} binary matrix (i.e., a matrix where each cell is either 0 or 1) such that its **parity matrix** is:
{parity}

**Definition (Parity Matrix):** For each cell (i, j), its parity is the XOR of the cell’s value and the values of its four neighbors (up, down, left, right). A neighbor outside the grid is treated as 0.

**Output Format:** Output {N} lines, each with {M} characters (each '0' or '1'), without separators. The format must match the input: one line per row."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the GridParityConstruction_Environment instance.
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

        one_probability = random.random()
        grid = ["".join("01"[random.random() < one_probability] for _ in range(M)) for _ in range(N)]
        self.parameter["reference_answer"] = "\n".join("".join(map(str, row)) for row in grid)

        parity = self.parameter["parity"] = [[0] * M for _ in range(N)]
        for i in range(N) :
            for j in range(M) :
                parity[i][j] ^= int(grid[i][j])
                for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)] :
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M :
                        parity[i][j] ^= int(grid[ni][nj])
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            parity = "\n".join("".join(map(str, row)) for row in self.parameter["parity"]),
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
            grid = processed_result
            
            if len(grid) != N or any(len(row) != M for row in grid) :
                return self.rewards["wrong_format"]
            for row in grid :
                if not all(c in "01" for c in row) :
                    return self.rewards["wrong_format"]
            
            parity = [[0] * M for _ in range(N)]
            for i in range(N) :
                for j in range(M) :
                    parity[i][j] ^= int(grid[i][j])
                    for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)] :
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M :
                            parity[i][j] ^= int(grid[ni][nj])
            
            satisfied = sum(int(parity[i][j] == self.parameter["parity"][i][j]) for i in range(N) for j in range(M))
            assert satisfied <= N * M, "satisfied should be less than or equal to N * M"

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (N * M)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == (N * M))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]