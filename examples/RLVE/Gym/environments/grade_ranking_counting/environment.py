import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GradeRankingCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3270
    prompt_template = \
r"""Count the number of matrices A of size {N} × {M} (0-indexed) that satisfy the following conditions:
1. Each element A[i][j] (0 ≤ i < {N}, 0 ≤ j < {M}) is an integer in the range [1, U[j]]. U is: {U}
2. For each column j (0 ≤ j < {M}), there are exactly R[j] rows i (1 ≤ i < {N}) such that A[i][j] > A[0][j]. R is: {R}
3. There are exactly {K} rows i (1 ≤ i < {N}) such that A[0][j] ≥ A[i][j] holds for **all** j (0 ≤ j < {M}).

Output the number of such matrices modulo {MOD}."""

    MODs = (10 ** 9 + 7, 998244353)

    def __init__(self,
                 wrong_format: float = -1.0, wrong_range: float = -0.5, correct_answer: float = +1.0, wrong_answer: float = 0.0,
                 **kwargs):
        """
        Initialize the GradeRankingCountingProblem instance.
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

        A = [[None] * M for _ in range(N)]
        losers = set(random.sample(range(1, N), k = random.randint(0, N - 1)))
        U = self.parameter["U"] = [random.randint(1, N) for j in range(M)]
        R = self.parameter["R"] = [0] * M
        for j in range(M) :
            A[0][j] = random.randint(1, U[j])
            for i in range(1, N) :
                if i in losers :
                    A[i][j] = random.randint(1, A[0][j])
                else :
                    A[i][j] = random.randint(1, U[j])
                R[j] += int(A[i][j] > A[0][j])
        K = self.parameter["K"] = sum(int(all(A[0][j] >= A[i][j] for j in range(M))) for i in range(1, N))
        assert K >= len(losers), "K should be at least the number of losers"


        # ---------- basic combinatorics ----------
        def prepare_factorials(limit: int):
            """pre-compute factorials and inverse factorials up to <limit> (inclusive)"""
            fact = [1] * (limit + 1)
            for i in range(1, limit + 1):
                fact[i] = fact[i - 1] * i % MOD
            inv_fact = [1] * (limit + 1)
            inv_fact[limit] = pow(fact[limit], MOD - 2, MOD)
            for i in range(limit, 0, -1):
                inv_fact[i - 1] = inv_fact[i] * i % MOD
            return fact, inv_fact


        def C(n: int, k: int) -> int:
            if k < 0 or k > n:
                return 0
            return FACT[n] * INV_FACT[k] % MOD * INV_FACT[n - k] % MOD


        # ---------- Σ k^p for huge k (Faulhaber via Lagrange) ----------
        def power_sum(p: int, n: int) -> int:
            """
            S_p(n) = Σ_{k=1..n} k^p   (0 ≤ p ≤ 2N ≈ 200, n may be 1e9)
            evaluated in O(p) with Lagrange interpolation over equally–spaced nodes 0 … p+1
            """
            if n == 0:
                return 0
            d = p + 1                     # degree of the polynomial
            if n <= d:                    # tiny n – direct loop is faster
                s = 0
                for k in range(1, n + 1):
                    s = (s + pow(k, p, MOD)) % MOD
                return s

            #   pre-compute y[i] = Σ_{k=1..i} k^p  for i = 0 … d   (0 ≤ d ≤ 200)
            y = [0] * (d + 1)
            partial = 0
            for i in range(1, d + 1):
                partial = (partial + pow(i, p, MOD)) % MOD
                y[i] = partial

            x = n % MOD

            #   total product P := Π_{j=0..d} (x − j)
            P = 1
            for j in range(d + 1):
                P = P * ((x - j) % MOD) % MOD

            #   Lagrange
            res = 0
            for i in range(d + 1):
                # numerator = P / (x - i)
                num = P * pow((x - i) % MOD, MOD - 2, MOD) % MOD

                # denominator = (-1)^{d-i} · i! · (d-i)!
                sign = MOD - 1 if (d - i) & 1 else 1
                denom_inv = sign * INV_FACT[i] % MOD * INV_FACT[d - i] % MOD

                res = (res + y[i] * num % MOD * denom_inv) % MOD
            return res


        # ---------- single course contribution ----------
        def course_contribution(U_i: int, A_i: int, N: int) -> int:
            """
            A_i students must be strictly above B in this course
            B_i = N-1-A_i students are ≤ B.
            f_i = Σ_{S=1..U_i} (U_i-S)^{A_i} · S^{B_i}
                = Σ_{j=0..A_i} (-1)^j C(A_i,j) U_i^{A_i-j} · Σ_{k=1..U_i} k^{B_i+j}
            """
            B_i = N - 1 - A_i
            V = U_i
            res = 0
            for j in range(A_i + 1):
                coeff = C(A_i, j)
                if j & 1:            # (-1)^j
                    coeff = MOD - coeff
                term = coeff * pow(V, A_i - j, MOD) % MOD
                term = term * power_sum(B_i + j, V) % MOD
                res = (res + term) % MOD
            return res


        # ---------- inclusion–exclusion over dominated students ----------
        def pattern_count(N: int, K: int, A_list):
            """
            Count ways to pick, for every course i, a subset of size A_i
            (taken from the S = N-1-K non-dominated students)
            so that every non-dominated student appears ≥1 time.
            """
            S = N - 1 - K
            total = 0
            for t in range(S + 1):           # t = number of non-dominated students *omitted*
                if t:   # early bailout for impossible A_i > S-t
                    ok = all(A <= S - t for A in A_list)
                    if not ok:
                        continue
                prod = 1
                for A in A_list:
                    prod = prod * C(S - t, A) % MOD
                term = C(S, t) * prod % MOD
                if t & 1:
                    total = (total - term) % MOD
                else:
                    total = (total + term) % MOD
            # finally multiply by ways to choose which K students are dominated
            total = total * C(N - 1, K) % MOD
            return total
        
        R = [r + 1 for r in R]
        # factorials up to 2N ≈ 200 cover everything (exponents ≤ 2N-1)
        MAX_F = 2 * N + 2
        FACT, INV_FACT = prepare_factorials(MAX_F)

        # per-course numeric factor f_i
        F_product = 1
        A_list = []
        for i in range(M):
            A_i = R[i] - 1       # number strictly above B
            A_list.append(A_i)
            F_product = F_product * course_contribution(U[i], A_i, N) % MOD

        # combinatorial patterns for the “> B” sets
        PATTERNS = pattern_count(N, K, A_list)

        answer = F_product * PATTERNS % MOD
        self.parameter["reference_answer"] = answer

    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            MOD = self.parameter["MOD"],
            K = self.parameter["K"],
            U = " ".join("U[{}]={}".format(j, Uj) for j, Uj in enumerate(self.parameter["U"])),
            R = " ".join("R[{}]={}".format(j, Rj) for j, Rj in enumerate(self.parameter["R"])),
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