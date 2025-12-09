import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumGCDWithIndividual_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4449
    prompt_template = \
r"""Please compute the sum of GCD(i, {N}) for all i such that 1 ≤ i ≤ {N}. Here, GCD(i, j) denotes the **greatest common divisor** of integers i and j.

**Output Format:** Your final answer should be a single integer indicating the sum of GCDs."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumGCDWithIndividual_Environment instance.
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
        assert MAX_N >= 4, "MAX_N should be greater than or equal to 4"

        N = self.parameter["N"] = random.randint(4, MAX_N)

        def f(n):
            t = n
            ans = n
            i = 2
            # iterate over possible prime factors up to sqrt(t), updating t as we go
            while i * i <= t:
                if t % i == 0:
                    b = 0
                    # count how many times i divides t
                    while t % i == 0:
                        b += 1
                        t //= i
                    # incorporate factor i with exponent b into ans
                    ans //= i
                    ans *= (b * i - b + i)
                i += 1

            # if there's any prime > sqrt(n) left
            if t > 1:
                ans //= t
                ans *= (t + t - 1)

            return ans
        self.parameter["reference_answer"] = f(N)
    

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