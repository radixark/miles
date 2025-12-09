import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class PairMoreOneCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3726
    prompt_template = \
r"""Please count the number of pairs of binary strings (S, T) such that:
- The length of S is {N} = {M} + {delta}, and the length of T is {M}.
- The number of 1s in S is strictly greater than the number of 1s in T.

Please output the result modulo 10^{K}."""


    def __init__(self,
                 max_K : int = 5,
                 wrong_format: float = -1.0, wrong_range: float = -0.5, correct_answer: float = +1.0, wrong_answer: float = 0.0,
                 **kwargs):
        """
        Initialize the PairMoreOneCountingProblem instance.
        """
        super().__init__(**kwargs)

        self.max_K = max_K
        assert self.max_K >= 1, "max_K must be at least 1"

        self.rewards = {
            "wrong_format": wrong_format,
            "wrong_range": wrong_range,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_M" in self.parameter, "MAX_M must be set in the parameter"
        MAX_M = self.parameter["MAX_M"]
        assert MAX_M >= 1, "MAX_M must be at least 1"

        assert "MAX_delta" in self.parameter, "MAX_delta must be set in the parameter"
        MAX_delta = self.parameter["MAX_delta"]
        assert MAX_delta >= 0, "MAX_delta must be at least 0"

        M = self.parameter["M"] = random.randint(1, MAX_M)
        delta = self.parameter["delta"] = random.randint(0, MAX_delta)
        N = M + delta

        K = self.parameter["K"] = random.randint(1, self.max_K)


        MOD10 = 10 ** K
        MOD2 = 2 ** (K + 1)
        MOD5 = 5 ** K
        MOD_ALL = MOD10 * 2  # = 2 * 10^K

        # Build factorial tables excluding factors of 2 and 5
        s2 = [1] * (MOD2 + 1)
        for i in range(1, MOD2 + 1):
            if i & 1 == 0:
                s2[i] = s2[i - 1]
            else:
                s2[i] = (s2[i - 1] * i) % MOD2

        s5 = [1] * (MOD5 + 1)
        for i in range(1, MOD5 + 1):
            if i % 5 == 0:
                s5[i] = s5[i - 1]
            else:
                s5[i] = (s5[i - 1] * i) % MOD5

        # Recursive factorial mod p^c excluding multiples of p
        def solve_fact(n, p, modp):
            if n <= 1:
                return 1
            sub = solve_fact(n // p, p, modp)
            if p == 2:
                sp_mod = s2[modp]
                sp_rem = s2[n % modp]
            else:
                sp_mod = s5[modp]
                sp_rem = s5[n % modp]
            return sub * pow(sp_mod, n // modp, modp) % modp * sp_rem % modp

        # Count exponent of p in n!
        def count_p(n, p):
            cnt = 0
            while n:
                n //= p
                cnt += n
            return cnt

        # Extended Lucas for C(n, m) mod 2*10^K
        def lucas(n, m):
            # 2-adic part
            c2 = count_p(n, 2) - count_p(m, 2) - count_p(n - m, 2)
            if c2 <= K:
                a2 = solve_fact(n, 2, MOD2)
                b2 = solve_fact(m, 2, MOD2)
                inv_b2 = pow(b2, -1, MOD2)
                a2 = a2 * inv_b2 % MOD2
                c2part = solve_fact(n - m, 2, MOD2)
                inv_c2 = pow(c2part, -1, MOD2)
                a2 = a2 * inv_c2 % MOD2 * pow(2, c2, MOD2) % MOD2
            else:
                a2 = 0

            # 5-adic part
            c5 = count_p(n, 5) - count_p(m, 5) - count_p(n - m, 5)
            if c5 < K:
                a5 = solve_fact(n, 5, MOD5)
                b5 = solve_fact(m, 5, MOD5)
                inv_b5 = pow(b5, -1, MOD5)
                a5 = a5 * inv_b5 % MOD5
                c5part = solve_fact(n - m, 5, MOD5)
                inv_c5 = pow(c5part, -1, MOD5)
                a5 = a5 * inv_c5 % MOD5 * pow(5, c5, MOD5) % MOD5
            else:
                a5 = 0

            # CRT combine (mod MOD2) = a2 and (mod MOD5) = a5
            t = (a5 - a2) * pow(MOD2, -1, MOD5) % MOD5
            return (a2 + MOD2 * t) % (MOD2 * MOD5)

        # Main computation
        if N == M:
            total = pow(2, 2 * N, MOD_ALL)
            comb = lucas(2 * N, N)
            ans = (total - comb) % MOD_ALL
            ans = (ans // 2) % MOD10
        else:
            total = pow(2, N + M, MOD_ALL)
            diff = N - M
            for i in range(1, diff):
                total = (total + lucas(N + M, M + i)) % MOD_ALL
            ans = (total // 2) % MOD10
        
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        M, delta = self.parameter["M"], self.parameter["delta"]
        return self.prompt_template.format(
            N = M + delta,
            M = M,
            delta = delta,
            K = self.parameter["K"],
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
            if not (0 <= processed_result < (10 ** self.parameter["K"])) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]