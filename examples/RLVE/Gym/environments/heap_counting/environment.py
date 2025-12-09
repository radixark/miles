import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class HeapCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2606
    prompt_template = r"""Compute the number of permutations `P` of the numbers 1 through {N} such that for all `2 ≤ i ≤ {N}`, it holds that `P[i] > P[i // 2]`. Since the answer may be large, output the result modulo {P}, where {P} is a prime number."""


    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the HeapCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"
        N = self.parameter["N"] = random.randint(3, MAX_N)

        is_prime = [True] * ((5 * N) + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int((5 * N) ** 0.5) + 1) :
            if is_prime[i] :
                for j in range(i * i, (5 * N) + 1, i):
                    is_prime[j] = False
        P = self.parameter["P"] = random.choice([i for i in range(2, (5 * N) + 1) if is_prime[i]])


        def mod_pow(a: int, b: int, p: int) -> int:
            """a^b mod p   with binary exponentiation"""
            res = 1
            while b:
                if b & 1:
                    res = res * a % p
                a = a * a % p
                b >>= 1
            return res


        def comb_small(n: int, k: int, p: int, fact: list) -> int:
            """C(n,k) mod p   with 0 ≤ n,k < p   (prime p)"""
            if k < 0 or k > n:
                return 0
            return fact[n] * mod_pow(fact[k] * fact[n - k] % p, p - 2, p) % p


        def lucas(n: int, k: int, p: int, fact: list) -> int:
            """C(n,k) mod p   prime p   via Lucas"""
            if k == 0:
                return 1
            return (lucas(n // p, k // p, p, fact) *
                    comb_small(n % p, k % p, p, fact)) % p

        # ---------- factorials mod P up to N ----------
        fact = [1] * (N + 1)
        for i in range(1, N + 1):
            fact[i] = fact[i - 1] * i % P

        # ---------- subtree sizes ----------
        S = [0] * (5 * N + 2)          # S[i] = size of subtree rooted at i
        for i in range(1, N + 1):
            S[i] = 1
        for i in range(N, 1, -1):       # bottom-up   (skip the root’s "parent")
            S[i >> 1] += S[i]

        # ---------- number of labellings ----------
        DP = [1] * (2 * N + 2)          # leaves already 1
        for i in range(N, 0, -1):
            left = i * 2
            right = left + 1
            dp_left = DP[left]          # 1 if child beyond n
            dp_right = DP[right]
            choose_left = lucas(S[i] - 1, S[left], P, fact)
            DP[i] = (choose_left * dp_left * dp_right) % P

        self.parameter["reference_answer"] = DP[1] % P


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], P = self.parameter["P"])


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
            if not (0 <= processed_result < self.parameter["P"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]