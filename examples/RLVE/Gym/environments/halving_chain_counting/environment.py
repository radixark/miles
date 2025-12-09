import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class HalvingChainCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1028
    prompt_template = \
r"""Construct sequences based on the following rules:

1. A sequence that contains only a single number `{N}` is considered a valid sequence.
2. Given any valid sequence, you can create a new valid sequence by appending a positive integer to the end — but the new number must be **at most half** of the last number in the current sequence (i.e., ≤ last_element / 2).

Your task is to determine how many **distinct valid sequences** can be constructed following these rules.

Output Format:
Your answer should be a single integer — the total number of valid sequences.
Example: `10` (do **NOT** include the backticks or quotes); this means there are 10 distinct valid sequences.
"""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the HalvingChainCounting_Environment instance.
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

        
        # Use dynamic programming to count the number of valid sequences
        dpF = [0] * (N + 1)
        for x in range(1, N + 1) :
            dpF[x] = 1
            for y in range(1, x // 2 + 1) :
                dpF[x] += dpF[y]
        self.parameter["reference_answer"] = dpF[N]
    
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