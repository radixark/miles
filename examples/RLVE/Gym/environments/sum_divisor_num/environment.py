import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Sum_DivisorNum_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3935
    prompt_template = \
r"""Please compute sum(d(i)) for all integers i such that {L} ≤ i ≤ {R}. Here, d(i) denotes the **number of positive divisors** of the integer i.

**Output Format:** Your final answer should be a single integer — the sum of d(i) over all i in the range [{L}, {R}]."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the Sum_DivisorNum_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "MAX_R" in self.parameter, "MAX_R is required in parameter"
        MAX_R = self.parameter["MAX_R"]
        assert MAX_R >= 2, "MAX_R should be greater than or equal to 2"

        R = self.parameter["R"] = random.randint(2, MAX_R)
        L = self.parameter["L"] = random.randint(1, R)
        assert 1 <= L <= R, "L should be less than or equal to R"


        def sumF(n : int) -> int :
            total = 0
            l = 1
            while l <= n :
                val = (n // l)
                r = n // (n // l)
                total += val * ((r - l + 1))
                l = r + 1
            return total
        self.parameter["reference_answer"] = sumF(R) - sumF(L - 1)
    
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