import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CapitalCityEffect_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3754
    prompt_template = r"""Let’s define f(x) as follows, where x is a positive integer in its base-10 representation:
- Divide x into **segments**, where each segment is a maximal substring consisting of the same digit.
- For each segment, compute `digit × (length of segment)^2`.
- Then, f(x) is the **sum** over all segments.
- For example, f(2334222) = 2×1² + 3×2² + 4×1² + 2×3² = 2 + 12 + 4 + 18 = 36, where the segments are `2` (length 1), `33` (length 2), `4` (length 1), and `222` (length 3).

Please output the sum of f(x) for all integers x in the range [{L}, {R}] (inclusive)."""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the CapitalCityEffect_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta
        }
    

    def _generate(self) -> None :
        assert "MAX_R" in self.parameter, "MAX_R is required in parameter"
        MAX_R = self.parameter["MAX_R"]
        assert MAX_R >= 20, "MAX_R should be greater than or equal to 20"
        R = self.parameter["R"] = random.randint(20, MAX_R)
        L = self.parameter["L"] = random.randint(1, R)


        def solve(x):
            digits = list(map(int, str(x)))
            n = len(digits)
            # memo for non-tight states: key = (pos, last, length, sum_), value = total houses
            dp = {}

            def dfs(pos, last, length, sum_, tight):
                # If we've placed all digits, add the final segment's contribution
                if pos == n:
                    return sum_ + (length * length * last if last != -1 else 0)

                # Only memoize when we're not tight
                if not tight:
                    key = (pos, last, length, sum_)
                    if key in dp:
                        return dp[key]

                maxd = digits[pos] if tight else 9
                ans = 0
                for d in range(maxd + 1):
                    if d == last:
                        # extend current segment
                        new_sum = sum_
                        new_len = length + 1
                    else:
                        # close off previous segment (if any) and start a new one
                        closed = (length * length * last) if last != -1 else 0
                        new_sum = sum_ + closed
                        new_len = 1
                    ans += dfs(pos + 1, d, new_len, new_sum, tight and d == maxd)

                if not tight:
                    dp[key] = ans
                return ans

            return dfs(0, -1, 0, 0, True)
        self.parameter["reference_answer"] = solve(R) - solve(L - 1)
    

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