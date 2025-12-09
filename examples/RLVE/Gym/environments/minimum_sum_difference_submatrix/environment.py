import random
from typing import Optional, List
from itertools import combinations
from Gym.environment import VerifiableEnvironment


class MinimumSumDifferenceSubmatrix_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a {N} × {M} matrix of integers (with row indices from `0` to `{N_minus_1}` and column indices from `0` to `{M_minus_1}`). Please select {R} rows and {C} columns, denoted as `r[1], ..., r[{R}]` and `c[1], ..., c[{C}]`, respectively, such that:
- 0 ≤ r[1] < ... < r[{R}] ≤ {N_minus_1}
- 0 ≤ c[1] < ... < c[{C}] ≤ {M_minus_1}

The matrix is given as below (each line represents a row):
{matrix}

From these, you can extract a new {R} × {C} submatrix, where the value at position `(i, j)` is taken from row `r[i]` and column `c[j]` of the original matrix. Try your best to **minimize the sum of absolute differences** between all pairs of **adjacent** (horizontally or vertically) elements in the new submatrix. Two elements are considered adjacent if their manhattan distance is 1 (i.e., they are either in the same row and consecutive columns, or in the same column and consecutive rows).

**Output Format:** Output two lines,
- The first line contains the selected row indices: `r[1], ..., r[{R}]`
- The second line contains the selected column indices: `c[1], ..., c[{C}]`
All integers in one line should be separated by a single space and should be **0-indexed** (i.e., the first row/column is `0`, the second is `1`, etc.)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize MinimumSumDifferenceSubmatrix_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(3, MAX_N_M), random.randint(3, MAX_N_M)
        R, C = self.parameter["R"], self.parameter["C"] = random.randint(2, N - 1), random.randint(2, M - 1)
        matrix = self.parameter["matrix"] = [[random.randint(1, N * M) for _ in range(M)] for _ in range(N)]


        # Compute an appropriate "infinite" value based on the input
        max_val = max(max(row) for row in matrix)
        # Maximum number of adjacent pairs in any R×C submatrix:
        # vertical: (R-1)*C, horizontal: R*(C-1)
        max_pairs = (R - 1) * C + R * (C - 1)
        INF = max_val * max_pairs + 1

        ans = INF

        # Enumerate all choices of R rows out of N
        for rows in combinations(range(N), R):
            # Precompute w[j][i]: the cost contribution when picking column j then column i
            # (and w[i][i] is the vertical adjacencies within column i)
            w = [[0] * M for _ in range(M)]

            for i in range(M):
                # Vertical adjacencies in column i
                for idx in range(1, R):
                    r0 = rows[idx - 1]
                    r1 = rows[idx]
                    w[i][i] += abs(matrix[r1][i] - matrix[r0][i])

                # Cross-column differences between column j and column i
                for j in range(i):
                    s = 0
                    for r0 in rows:
                        s += abs(matrix[r0][i] - matrix[r0][j])
                    w[j][i] = s

            # DP over columns: dp[i][k] = min cost to pick k columns ending at column i
            dp = [[INF] * (C + 1) for _ in range(M)]
            for i in range(M):
                dp[i][1] = w[i][i]

            for k in range(2, C + 1):
                for i in range(M):
                    best = INF
                    for j in range(i):
                        cost = dp[j][k - 1] + w[j][i] + w[i][i]
                        if cost < best:
                            best = cost
                    dp[i][k] = best

            # Update global answer
            for i in range(M):
                if dp[i][C] < ans:
                    ans = dp[i][C]

        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            M = M,
            M_minus_1 = M - 1,
            R = self.parameter["R"],
            C = self.parameter["C"],
            matrix = "\n".join(" ".join(map(str, row)) for row in self.parameter["matrix"]),
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

            if len(processed_result) != 2 :
                return self.rewards["wrong_format"]
            row_indices, col_indices = processed_result
            if len(row_indices) != self.parameter["R"] or len(col_indices) != self.parameter["C"] :
                return self.rewards["wrong_format"]
            
            if not all(0 <= row < self.parameter["N"] for row in row_indices) or not all(0 <= col < self.parameter["M"] for col in col_indices) :
                return self.rewards["invalid_solution"]
            if not all(row_indices[i] < row_indices[i + 1] for i in range(len(row_indices) - 1)) or not all(col_indices[i] < col_indices[i + 1] for i in range(len(col_indices) - 1)) :
                return self.rewards["invalid_solution"]
            
            new_matrix = [[self.parameter["matrix"][row][col] for col in col_indices] for row in row_indices]
            sum_diff = 0
            for i in range(self.parameter["R"]):
                for j in range(self.parameter["C"]):
                    if i < self.parameter["R"] - 1:
                        sum_diff += abs(new_matrix[i + 1][j] - new_matrix[i][j])
                    if j < self.parameter["C"] - 1:
                        sum_diff += abs(new_matrix[i][j + 1] - new_matrix[i][j])
            gold, answer = self.parameter["gold_answer"], sum_diff
            assert gold <= answer

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "gold should be 0 if answer is 0"
                    return self.rewards["rewarding_weight"]
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]