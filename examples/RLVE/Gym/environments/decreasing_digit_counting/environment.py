import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DecreasingDigitCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1066
    prompt_template = \
r"""Let R be a number in base 2^{K} = {power_2_K}, satisfying the following conditions:
- R must be **at least a 2-digit** number in base 2^{K} (leading zeros are ignored; i.e., we don’t count numbers like `01` or `0005`).
- When viewed as a number in base 2^{K}, each digit of R, except for the last one, must be **strictly less than** its immediate right neighbor. (Digits are read from **left to right**, with the leftmost digit being the most significant — following natural reading order.)
- When R is converted to its binary representation, the total number of bits (ignoring leading zeros) must not exceed {W}.

Your task is to determine how many **distinct valid values of R** satisfy all the above conditions.

**Output Format:**  
Your final answer should be a single integer — the total number of distinct values of R.  
Example: `10` (do **NOT** include the backticks or quotes); this means there are 10 valid values of R that satisfy the conditions.
"""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the NumberPartitionCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 2, "MAX_K should be greater than or equal to 2"

        assert "MAX_W" in self.parameter, "MAX_W is required in parameter"
        MAX_W = self.parameter["MAX_W"]
        assert MAX_W >= 1, "MAX_W should be greater than or equal to 1"
        
        K = self.parameter["K"] = random.randint(2, MAX_K)
        W = self.parameter["W"] = random.randint(K + 1, min(MAX_W, K * (1 << K))) if K + 1 <= min(MAX_W, K * (1 << K)) else MAX_W


        r0 = W % K
        m_max = W // K + (1 if r0 != 0 else 0)
        if m_max < 2 :
            answer = 0
        else :
            max_val = (1 << K) - 1
            total = 0
            for m in range(2, m_max + 1) :
                if m > max_val :
                    continue
                if m < m_max or (m == m_max and r0 == 0) :
                    total += math.comb(max_val, m)
                else :
                    max_high = (1 << r0) - 1
                    for i in range(1, max_high + 1) :
                        ni = max_val - i
                        mi = m - 1
                        if ni >= mi :
                            total += math.comb(ni, mi)
            answer = total
        self.parameter["reference_answer"] = answer
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(K = self.parameter["K"], W = self.parameter["W"], power_2_K = 2 ** self.parameter["K"])


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