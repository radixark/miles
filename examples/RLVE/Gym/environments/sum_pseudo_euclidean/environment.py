import random
from typing import Optional
from Gym.environment import VerifiableEnvironment

class SumPseudoEuclidean_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3764
    prompt_template = \
r"""Consider the function `f(a, b)` defined in Python as follows:
```python
def f(a: int, b: int) -> int:
    if a == b:
        return 0
    if a > b:
        return f(a - b, b + b) + 1
    else:
        return f(a + a, b - a) + 1
```

If the function enters an infinite loop, we treat its return value as `0`. Tell me the sum of `f(i, j)` over all pairs (i, j) such that 1 ≤ i ≤ {N} and 1 ≤ j ≤ {N}."""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumPseudoEuclidean_Environment instance.
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
        assert MAX_N >= 5, "MAX_N should be greater than or equal to 5"

        N = self.parameter["N"] = random.randint(5, MAX_N)


        def solve(N):
            # Count of odd numbers in [x, y]
            def count_odds(x, y):
                length = y - x + 1
                # If the interval length is odd and starts with an odd number, we get one extra odd
                if (length & 1) and (x & 1):
                    return (length >> 1) + 1
                else:
                    return length >> 1

            # “Logarithmic” number‐theory block over [l..k]
            def block_sum(l, k, N):
                total = 0
                while l <= k:
                    # floor(log2(l))
                    lg = l.bit_length() - 1
                    # r = min((2^(lg+1) - 1), k)
                    r = min((1 << (lg + 1)) - 1, k)
                    # contribution: lg * (N//l) times number of odds in [l..r]
                    total += lg * (N // l) * count_odds(l, r)
                    l = r + 1
                return total

            ans = 0
            l = 1
            # Standard divisor‐block decomposition over 1..N
            while l <= N:
                v = N // l
                r = N // v
                ans += block_sum(l, r, N)
                l = r + 1

            # multiply by 2 as in the original C++ (ans << 1)
            return ans * 2

        self.parameter["reference_answer"] = solve(N)
        assert self.parameter["reference_answer"] > 0, "Reference answer should be greater than 0"
    

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