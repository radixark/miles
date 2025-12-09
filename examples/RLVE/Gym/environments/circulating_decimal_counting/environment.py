from math import gcd
from functools import lru_cache
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CirculatingDecimalCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1587
    prompt_template = \
r"""Please count how many **distinct pure repeating decimals** (in terms of numeric value) exist in base ${K}$, that can be written as a reduced fraction $\frac{x}{y}$ where $1 \le x \le {N}$ and $1 \le y \le {M}$, with $x$ and $y$ being integers.
A number is called a **pure repeating decimal** if and only if it can be written in the form of $$a.\dot{c_1} c_2 c_3 \dots c_{p - 1} \dot{c_p}$$, where $a$ is an integer, $p \ge 1$, and each $c_i$ ($1 \le i \le p$) is a digit in base ${K}$.

Examples:
- In base 10, $0.454545\ldots = 0.\dot{4}\dot{5}$ is a pure repeating decimal; it can be written as $\frac{5}{11}$ or $\frac{10}{22}$.
- In contrast, $0.166666\ldots = 0.1\dot{6}$ is **not** pure repeating in base 10; it can be written as $\frac{1}{6}$.

Note:
- **Integers are considered pure repeating**, because their decimal part can be represented as a repeating sequence of 0s.
- **Finite decimals with non-zero fractional parts** are **not** considered pure repeating.

**Output Format:** Your final answer should be a single integer — the total number of such distinct pure repeating decimals."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the CirculatingDecimalCounting_Environment instance.
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
        assert MAX_N >= 1, "MAX_N should be greater than or equal to 1"
        N = self.parameter["N"] = random.randint(1, MAX_N)

        assert "MAX_M" in self.parameter, "MAX_M is required in parameter"
        MAX_M = self.parameter["MAX_M"]
        assert MAX_M >= 1, "MAX_M should be greater than or equal to 1"
        M = self.parameter["M"] = random.randint(1, MAX_M)

        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 2, "MAX_K should be greater than or equal to 2"
        K = self.parameter["K"] = random.randint(2, MAX_K)


        LIM = min(M, max(K, int(M ** 0.5) + 1))

        g = [0] * (K + 1)
        for i in range(1, K + 1):
            g[i] = g[i - 1] + (1 if gcd(i, K) == 1 else 0)

        mu = [0] * (LIM + 1)
        is_comp = [False] * (LIM + 1)
        f = [0] * (LIM + 1)
        primes = []

        mu[1] = 1
        f[1] = 1

        def G(x):
            return (x // K) * g[K] + g[x % K]

        for i in range(2, LIM + 1):
            if not is_comp[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                ip = i * p
                if ip > LIM:
                    break
                is_comp[ip] = True
                if i % p == 0:
                    mu[ip] = 0
                    break
                else:
                    mu[ip] = -mu[i]
            f[i] = f[i - 1] + mu[i] * (G(i) - G(i - 1))

        @lru_cache(None)
        def F(x):
            if x <= LIM:
                return f[x]
            res = 1
            l = 2
            while l <= x:
                t = x // l
                r = x // t
                res -= F(t) * (G(r) - G(l - 1))
                l = r + 1
            return res

        ans = 0
        l = 1
        up = min(N, M)
        while l <= up:
            n_div = N // l
            m_div = M // l
            r = min(N // n_div, M // m_div)
            ans += n_div * G(m_div) * (F(r) - F(l - 1))
            l = r + 1

        assert ans > 0
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.replace(r"{K}", str(self.parameter["K"])) \
                                  .replace(r"{N}", str(self.parameter["N"])) \
                                  .replace(r"{M}", str(self.parameter["M"]))
    

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