import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class NegativeBase_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1017
    prompt_template = \
r"""We can represent integers using a **negative base** system with base `-R`, where `R` is a positive integer greater than 1. In this system, the digits used are from `0` to `R - 1` (in decimal).
For example, the decimal number `-15` can be represented as `110001` in base `-2`, since:
1×(-2)^5 + 1×(-2)^4 + 0×(-2)^3 + 0×(-2)^2 + 0×(-2)^1 + 1×(-2)^0 = (-15).

Your task is to convert the decimal number `{N}` into base `-{R}`, and output its digits (in decimal) from most significant to least significant.

Output Format:
Your final answer should be a single line containing the digits (in decimal), separated by **spaces**.
Example: `{R_minus_1} 0 1` (do **NOT** include the backticks or quotes) means `{R_minus_1} * (-{R})^2 + 0 * (-{R})^1 + 1 * (-{R})^0` in decimal.
"""

    def __init__(self,
                 wrong_format : float = -1.0, wrong_length : float = 0.0, rewarding_strategy : str = "mean([gold=answer])", rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the NegativeBase_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_length" : wrong_length,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 1, "MAX_N should be greater than or equal to 1"

        assert "MAX_R" in self.parameter, "MAX_R is required in parameter"
        MAX_R = self.parameter["MAX_R"]
        assert MAX_R >= 2, "MAX_R should be greater than or equal to 2"

        N = 0
        while N == 0 :
            N = self.parameter["N"] = random.randint(-MAX_N, MAX_N)
        R = self.parameter["R"] = random.randint(2, MAX_R)

        # Convert N to base -R
        def convert_to_negative_base(n, r) :
            if n == 0 :
                return []
            m = n % r
            if m < 0 :
                m -= r
                n += r
            return convert_to_negative_base(n // r, r) + [m]
        self.parameter["gold_answer"] = convert_to_negative_base(N, -R)
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))

        # check the gold_answer
        Sum = 0
        for digit in self.parameter["gold_answer"] :
            Sum *= (-R)
            Sum += digit
        assert Sum == N, "Sum should be equal to N, but got {} != {}".format(Sum, N)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            R = self.parameter["R"],
            R_minus_1 = self.parameter["R"] - 1,
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"
            if len(processed_result) != len(self.parameter["gold_answer"]) :
                return self.rewards["wrong_length"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])" :
                return self.rewards["rewarding_weight"] * (sum(float(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / len(self.parameter["gold_answer"]))
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * all(a == b for a, b in zip(self.parameter["gold_answer"], processed_result))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]