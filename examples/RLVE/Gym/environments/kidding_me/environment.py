import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class KiddingMe_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3266
    prompt_template = \
r"""Please compute the number of {N} × {M} matrices X, such that:
- For each 1 <= i <= {N}, 1 <= j <= {M}, we have 0 <= X[i][j] <= {M}
- For each 1 <= i <= {N}, 1 <= j < {M}, we have X[i][j] < X[i][j + 1]
- For each 1 < i <= {N}, 1 <= j < {M}, we have X[i][j] < X[i - 1][j + 1]

Please output the result module {MOD}
"""

    MODs = (10 ** 9 + 7, 998244353)

    def __init__(self,
                 wrong_format: float = -1.0, wrong_range: float = -0.5, correct_answer: float = +1.0, wrong_answer: float = 0.0,
                 **kwargs):
        """
        Initialize the KiddingMe_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "wrong_range": wrong_range,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)
        MOD = self.parameter["MOD"] = random.choice(self.MODs)


        # ---------- pre-compute factorials and inverse factorials ----------
        UP = max(N, M) * 3 + 5            # safe upper bound for every x + y that appears
        inv = [0] * (UP + 1)              # modular inverses of 1 … UP
        fact = [1] * (UP + 1)             # factorials
        inv_fact = [1] * (UP + 1)         # inverse factorials (1 / k!)

        inv[1] = 1
        for i in range(2, UP + 1):
            inv[i] = MOD - MOD // i * inv[MOD % i] % MOD

        for i in range(1, UP + 1):
            fact[i] = fact[i - 1] * i % MOD
            inv_fact[i] = inv_fact[i - 1] * inv[i] % MOD

        # ---------- helpers ----------
        def comb(x: int, y: int) -> int:
            """C(x + y, x) under MOD (return 0 if any index is negative)."""
            if x < 0 or y < 0:
                return 0
            return fact[x + y] * inv_fact[x] % MOD * inv_fact[y] % MOD


        def flip1(x: int, y: int) -> tuple[int, int]:
            """Perform the first reflection: swap, then (x--, y++)."""
            return y - 1, x + 1


        def flip2(x: int, y: int) -> tuple[int, int]:
            """Perform the second reflection: swap, then (x += M + 2, y -= M + 2)."""
            return y + M + 2, x - (M + 2)


        # ---------- main inclusion–exclusion ----------
        x, y = N + M + 1, N
        ans = comb(x, y)

        while x >= 0 and y >= 0:
            x, y = flip1(x, y)
            ans = (ans - comb(x, y)) % MOD
            x, y = flip2(x, y)
            ans = (ans + comb(x, y)) % MOD

        x, y = N + M + 1, N
        while x >= 0 and y >= 0:
            x, y = flip2(x, y)
            ans = (ans - comb(x, y)) % MOD
            x, y = flip1(x, y)
            ans = (ans + comb(x, y)) % MOD

        # ---------- output ----------
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
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