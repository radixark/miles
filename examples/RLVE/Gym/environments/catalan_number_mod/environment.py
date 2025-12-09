import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CatalanNumberMod_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3200
    prompt_template = \
r"""We define a **valid permutation** of the integers from 1 to 2×{N} (i.e., a permutation A[1], A[2], ..., A[2×{N}]) that satisfies all of the following conditions:
- A[1] < A[3] < ... < A[2×{N} - 1] (all elements at odd indices form a strictly increasing sequence)
- A[2] < A[4] < ... < A[2×{N}] (all elements at even indices form a strictly increasing sequence)
- For all i = 1 to {N}, A[2i - 1] < A[2i] (each adjacent pair forms an increasing pair)

Please compute the total number of such valid permutations. Output the result modulo {MOD}."""

    def __init__(self,
                 max_MOD : int = 1000000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the CatalanNumberMod_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 2, "MAX_N should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N)
        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        limit = 2 * N

        # Linear sieve to compute smallest prime factor (spf) for each number up to 2N
        spf = [0] * (limit + 1)
        primes = []
        for i in range(2, limit + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
            for p in primes:
                ip = i * p
                if p > spf[i] or ip > limit:
                    break
                spf[ip] = p

        # cnt[i] will hold the exponent contribution of i in the product:
        #   numerator: product of (n+2)*(n+3)*...*(2n)
        #   denominator: product of 1*2*...*n
        cnt = [0] * (limit + 1)
        # subtract denominator
        for i in range(1, N + 1):
            cnt[i] = -1
        # add numerator (skip N+1, since it's neither in numerator nor denominator)
        for i in range(N + 2, limit + 1):
            cnt[i] = 1

        # Propagate those counts down to prime factors
        for i in range(limit, 1, -1):
            if spf[i] < i:
                c = cnt[i]
                cnt[spf[i]] += c
                cnt[i // spf[i]] += c

        # Multiply out primes^cnt[p] mod P
        result = 1
        for p in primes:
            exp = cnt[p]
            if exp:
                result = result * pow(p, exp, MOD) % MOD

        self.parameter["reference_answer"] = result
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], MOD = self.parameter["MOD"])
    

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