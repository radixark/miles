import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class PowerCycle_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1050
    prompt_template = \
r"""It is well known that the **last digit** of positive powers of 2 follows a repeating pattern:
`2, 4, 8, 6, 2, 4, 8, 6, ...`.
We say that the **last digit** of powers of 2 has a **cycle length of 4** (there are other cycle lengths, but we focus only on the **smallest** one).

Now, your task is to analyze powers of a given integer {N} and determine whether the **last {K} digits** (in base-10) of its positive powers form a repeating cycle. If so, what is the **minimum** cycle length?

**Important Notes:**
1. If a power of {N} has fewer than {K} digits, consider the **missing leading digits as 0** (i.e., pad with zeros from the left).
2. If the cycle length is `L`, it means for **every positive integer** `a`, the last {K} digits of `{N}^a` are the same as those of `{N}^(a+L)`.

**Output Format:**
Your final answer should be a single integer representing the minimum cycle length.
Example: `10` (do **NOT** include the backticks or quotes)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = 0.0, rewarding_strategy : str = "gold/answer", rewarding_weight : float = 1.0,
                 **kwargs) :
        """
        Initialize the PowerCycle_Environment instance.
        """

        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "digit_num" in self.parameter, "digit_num is required in parameter"
        digit_num = self.parameter["digit_num"]
        assert digit_num >= 1, "digit_num should be greater than or equal to 1"

        def solve(S, K) :
            mod = 10 ** K
            # t is the original number mod 10^k.
            t = S % mod

            # Initially, last (which we use as the multiplier seed) equals t.
            last = t
            # ans will accumulate the cycle length.
            ans = 1
            # n_val will hold the intermediate product that we compare with t.
            n_val = t

            # For each digit position from 1 to k (i.e. considering the last i digits)
            for i in range(1, K + 1) :
                _last = 1
                flag = False
                # Try multipliers j = 1 to 10.
                for j in range(1, 11) :
                    # Update n_val and _last using multiplication mod 10^k.
                    n_val = (n_val * last) % mod
                    _last = (_last * last) % mod
                    # Compare the last i digits:
                    # This is done by comparing n_val mod 10^i with t mod 10^i.
                    if n_val % (10 ** i) == t % (10 ** i) :
                        # If j is less than 10, use j; otherwise, use 10.
                        multiplier = j if j < 10 else 10
                        ans *= multiplier
                        flag = True
                        break
                # If no valid multiplier was found in [1, 10], there is no cycle.
                if not flag :
                    return -1
                # Reset n_val for the next outer iteration.
                n_val = t
                # Set last to _last so that the cycle for the next higher digit is built on
                last = _last

            return ans
        
        while True :
            self.parameter["N"] = random.randint(1, 10 ** digit_num - 1)
            self.parameter["K"] = random.randint(1, digit_num)
            self.parameter["reference_answer"] = solve(self.parameter["N"], self.parameter["K"])

            if self.parameter["reference_answer"] != -1 :
                break
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"])
    

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
            assert self.parameter["reference_answer"] > 0, "reference_answer should be greater than 0"
            
            if self.rewards["rewarding_strategy"] == "gold/answer" :
                if processed_result % self.parameter["reference_answer"] == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] / processed_result)
                else :
                    return self.rewards["invalid_answer"]
            elif self.rewarding_strategy == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]