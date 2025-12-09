import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumProductDivisorNum_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3327
    prompt_template = \
r"""Please compute sum(d(i * j)) for all pairs (i, j) such that 1 ≤ i ≤ {N} and 1 ≤ j ≤ {M}. Here, d(x) denotes the **number of distinct divisors** of integer x, and d(i * j) is the number of divisors of the product of i and j.

**Output Format:** Your final answer should be a single integer — the sum of d(i * j) over all such pairs."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumProductDivisorNum_Environment instance.
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

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)


        def precompute(max_val: int):
            """
            Pre-computes
            • mu_pref[x]  – Σ_{k=1..x} μ(k)     (Möbius prefix sum, 0-indexed)
            • s[x]        – Σ_{k=1..x} ⌊x/k⌋   (harmonic-sum helper), 0-indexed
            Both lists have length max_val + 1 so that index == argument.
            """
            # -------- linear sieve for Möbius -----------------
            mu = [0] * (max_val + 1)           # μ itself; will turn into prefix sum
            mu[1] = 1
            is_composite = [False] * (max_val + 1)
            primes = []

            for i in range(2, max_val + 1):
                if not is_composite[i]:          # i is prime
                    primes.append(i)
                    mu[i] = -1
                for p in primes:
                    ip = i * p
                    if ip > max_val:
                        break
                    is_composite[ip] = True
                    if i % p == 0:               # p divides i  → μ(ip) = 0
                        mu[ip] = 0
                        break
                    mu[ip] = -mu[i]

            # turn μ into its prefix sum in-place
            for i in range(1, max_val + 1):
                mu[i] += mu[i - 1]

            # -------- pre-compute s[x] = Σ_{k=1..x} ⌊x/k⌋ -----
            s = [0] * (max_val + 1)
            for x in range(1, max_val + 1):
                res = 0
                i = 1
                # harmonic-series blocking: next j s.t. ⌊x/i⌋ is constant on [i,j]
                while i <= x:
                    j = x // (x // i)            # largest j with ⌊x/i⌋ constant
                    res += (j - i + 1) * (x // i)
                    i = j + 1
                s[x] = res

            return mu, s


        def solve_case(N: int, M: int, mu_pref, s):
            """
            Computes Σ_{i=1..N} Σ_{j=1..M} d(i j) in O(√(min(N,M))) using the
            Möbius inversion trick exactly as in the reference C++.
            N ≤ M must hold when called.
            """
            ans = 0
            i = 1
            while i <= N:
                j = min(N // (N // i), M // (M // i))
                ans += (mu_pref[j] - mu_pref[i - 1]) * s[N // i] * s[M // i]
                i = j + 1
            return ans

        # one-shot pre-computation up to the largest N, M
        mu_pref, s = precompute(max(N, M))

        if N > M:          # ensure N ≤ M as in the C++ optimisation
            N, M = M, N
        self.parameter["reference_answer"] = solve_case(N, M, mu_pref, s)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"])


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
            if processed_result <= 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]