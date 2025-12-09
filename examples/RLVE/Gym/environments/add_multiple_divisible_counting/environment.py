import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class AddMultiple_Divisible_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4466
    prompt_template = \
r"""Please compute the number of pairs (a, b) such that:
- 1 ≤ a < b ≤ {N}
- a × b is divisible by a + b

**Output Format:** Your final answer should be a single integer — the number of such pairs (a, b)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the AddMultiple_Divisible_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 6, "MAX_N should be greater than or equal to 6"

        N = self.parameter["N"] = random.randint(6, MAX_N)


        def calc(x : int, y : int) -> int :
            """
            Compute
                sum_{k = x+1..2*x-1} floor(y / k)
            by grouping k’s with the same quotient.
            """
            if y == 0 :
                return 0
            a = 0
            z = x << 1
            i = x + 1
            while i < z :
                q = y // i
                if q == 0 :
                    break
                j = min(y // q, z - 1)
                a += (j - i + 1) * q
                i = j + 1
            return a

        m = math.isqrt(N)

        mu = [0] * (m + 1)
        mu[1] = 1
        is_comp = [False] * (m + 1)
        primes = []

        for i in range(2, m + 1) :
            if not is_comp[i] :
                primes.append(i)
                mu[i] = -1
            for p in primes :
                ip = i * p
                if ip > m :
                    break
                is_comp[ip] = True
                if i % p == 0 :
                    mu[ip] = 0
                    break
                else :
                    mu[ip] = -mu[i]

        ans = 0
        for i in range(1, m + 1) :
            if mu[i] == 0 :
                continue
            ii = i * i
            top = m // i
            for j in range(1, top + 1) :
                y = N // (ii * j)
                ans += mu[i] * calc(j, y)
        assert ans > 0, "Answer should be greater than 0"
        self.parameter["reference_answer"] = ans
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])


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