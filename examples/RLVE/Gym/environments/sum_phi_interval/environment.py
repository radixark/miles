import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumPHIInterval_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3601
    prompt_template = r"""Define F(x) as the number of integers in the range [1, x] that are **not coprime** to x. Please output the sum of F(i) for all integers i in the range [{L}, {R}] (inclusive)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumPHIInterval_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_DELTA" in self.parameter, "MAX_DELTA is required in parameter"
        MAX_DELTA = self.parameter["MAX_DELTA"]
        assert MAX_DELTA >= 1, "MAX_DELTA should be greater than or equal to 1"

        L = self.parameter["L"] = random.randint(1, MAX_DELTA ** 2)
        R = self.parameter["R"] = L + random.randint(1, MAX_DELTA)


        # 1. generate all primes up to sqrt(R)
        limit = math.isqrt(R)
        is_prime = [True] * (limit + 1)
        primes = []
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                if i * i <= limit:
                    for j in range(i * i, limit + 1, i):
                        is_prime[j] = False

        # 2. prepare A and B arrays for [L..R]
        size = R - L + 1
        A = [L + i for i in range(size)]   # will become φ(L+i)
        B = [L + i for i in range(size)]   # copy to strip prime factors

        # 3. for each small prime p, apply the φ‐factor and strip p from B
        for p in primes:
            if p * p > R:
                break
            # first multiple of p in [L..R]
            start = ((L + p - 1) // p) * p
            for x in range(start, R + 1, p):
                idx = x - L
                # multiply φ‐part: φ(n) *= (1 - 1/p)
                A[idx] //= p
                A[idx] *= (p - 1)
                # remove ALL factors of p from B[idx]
                while B[idx] % p == 0:
                    B[idx] //= p

        # 4. any B[idx] > 1 is a leftover prime > sqrt(R)
        ans = 0
        for i in range(size):
            if B[i] > 1:
                # apply its φ‐factor
                A[i] //= B[i]
                A[i] *= (B[i] - 1)
            # qiandao(L+i) = (L+i) - φ(L+i)
            ans += (L + i) - A[i]

        self.parameter["reference_answer"] = ans
        assert ans > 0, "The reference answer should be greater than 0"
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(L = self.parameter["L"], R = self.parameter["R"])
    

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
            if processed_result < 0 :
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