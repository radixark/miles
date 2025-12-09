import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MatrixRMQCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3813
    prompt_template = \
r"""Count the number of matrices `A` of size {H} × {W} (1-indexed, meaning row indices range from 1 to {H} and column indices from 1 to {W}) such that:
1. Each element of `A` is an integer between 1 and {M}, inclusive.
2. The matrix satisfies the following {N} constraints, where `max(A[x1 : x2 + 1, y1 : y2 + 1])` denotes the maximum value in the contiguous submatrix defined by the corners (x1, y1) and (x2, y2) (inclusive):
{constraints}

Output a single integer — the number of such matrices modulo {MOD}."""
    def __init__(self,
                 H_W_range : int = 2,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MatrixRMQCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.H_W_range = H_W_range
        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        H = self.parameter["H"] = random.randint(1, N * self.H_W_range)
        W = self.parameter["W"] = random.randint(1, N * self.H_W_range)
        M = self.parameter["M"] = random.randint(1, (N * self.H_W_range) ** 2)

        A = [[random.randint(1, M) for _ in range(W)] for _ in range(H)]
        self.parameter["constraints"] = constraints = []
        for _ in range(N) :
            row_length, col_length = random.randint(1, H), random.randint(1, W)
            x1 = random.randint(1, H - row_length + 1)
            y1 = random.randint(1, W - col_length + 1)
            x2, y2 = x1 + row_length - 1, y1 + col_length - 1
            v = max(A[i - 1][j - 1] for i in range(x1, x2 + 1) for j in range(y1, y2 + 1))
            constraints.append((x1, y1, x2, y2, v))
        
        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)
        

        pos = []
        X = [1, H + 1]
        Y = [1, W + 1]
        # Read constraints and collect coordinates for compression
        for x1, y1, x2, y2, v in constraints:
            assert 1 <= x1 <= x2 <= H, "Invalid x1, x2 range"
            assert 1 <= y1 <= y2 <= W, "Invalid y1, y2 range"
            assert 1 <= v <= M, "Invalid value v"
            # include x2+1, y2+1 as open intervals
            pos.append((x1, y1, x2 + 1, y2 + 1, v))
            X.append(x1)
            X.append(x2 + 1)
            Y.append(y1)
            Y.append(y2 + 1)
        # Coordinate compression
        X = sorted(set(X))
        Y = sorted(set(Y))
        xi = {x: i for i, x in enumerate(X)}
        yi = {y: i for i, y in enumerate(Y)}
        # Precompute block ranges for each constraint
        ranges = []
        for x1, y1, x2p, y2p, v in pos:
            xl = xi[x1]
            xr = xi[x2p]
            yl = yi[y1]
            yr = yi[y2p]
            ranges.append((xl, xr, yl, yr, v))
        # Number of blocks in compressed grid
        Wb = len(X) - 1
        Hb = len(Y) - 1
        ans = 0
        # Inclusion-exclusion over subsets of constraints
        for mask in range(1 << N):
            # Initialize each block's max allowed value to M
            arr = [[M] * Hb for __ in range(Wb)]
            # Apply each constraint, reducing allowed max by 1 if in the subset
            for j in range(N):
                bit = (mask >> j) & 1
                xl, xr, yl, yr, v = ranges[j]
                limit = v - bit
                for xi_ in range(xl, xr):
                    row = arr[xi_]
                    for yi_ in range(yl, yr):
                        if row[yi_] > limit:
                            row[yi_] = limit
            # Compute number of fillings for this configuration
            tmp = 1
            for xi_ in range(Wb):
                dx = X[xi_ + 1] - X[xi_]
                for yi_ in range(Hb):
                    dy = Y[yi_ + 1] - Y[yi_]
                    area = dx * dy
                    val = arr[xi_][yi_]
                    # pow handles zero and mod efficiently
                    tmp = tmp * pow(val, area, MOD) % MOD
                    if tmp == 0:
                        break
                if tmp == 0:
                    break
            # Inclusion-exclusion sign
            if bin(mask).count('1') & 1:
                ans = (ans - tmp) % MOD
            else:
                ans = (ans + tmp) % MOD
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            H = self.parameter["H"],
            W = self.parameter["W"],
            M = self.parameter["M"],
            N = self.parameter["N"],
            constraints = "\n".join("max(A[{} : {} + 1, {} : {} + 1]) = {}".format(x1, x2, y1, y2, v) for x1, y1, x2, y2, v in self.parameter["constraints"]),
            MOD = self.parameter["MOD"],
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
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]