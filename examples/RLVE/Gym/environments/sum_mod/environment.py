import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumMOD_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2260
    prompt_template = \
r"""Please compute the sum of ({N} mod i) × ({M} mod j) over all pairs of integers (i, j) such that:
- 1 ≤ i ≤ {N}
- 1 ≤ j ≤ {M}
- i ≠ j

**Output Format:** Your final answer should be a single integer — the sum of all computed values."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumMOD_Environment instance.
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
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N_M)
        M = self.parameter["M"] = random.randint(3, MAX_N_M)


        def sum1(l : int, r : int) -> int :
            return (l + r) * (r - l + 1) // 2

        def sum2(x : int) -> int :
            return x * (x + 1) * (2 * x + 1) // 6

        def calc(n : int) -> int :
            res, l = 0, 1
            while l <= n :
                q = n // l
                r = n // q
                res += n * (r - l + 1) - sum1(l, r) * q
                l = r + 1
            return res

        def solve(n : int, m : int) -> int :
            if n > m :
                n, m = m, n

            ans = calc(n) * calc(m)

            l = 1
            while l <= n :
                nd, md = n // l, m // l
                r = min(n // nd, m // md)

                cnt  = r - l + 1
                SUM  = n * m * cnt
                Sum  = nd * md * (sum2(r) - sum2(l - 1))
                SUMK = (nd * m + md * n) * sum1(l, r)
                ans -= (SUM + Sum - SUMK)
                l = r + 1

            return ans

        self.parameter["reference_answer"] = solve(N, M)

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
            if processed_result < 0 :
                return self.rewards["wrong_format"]
            
            if self.parameter["reference_answer"] == 0 :
                return self.rewards["rewarding_weight"] * (processed_result == 0)

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]