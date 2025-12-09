import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class RootExtraction_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Your task is to compute the **{K}-th root of {N}**, that is, find the value of `{N}^(1/{K})`.

Since the result may not be an exact integer, output the value in **decimal form**, as accurate as possible, **up to 5 decimal places**.
If the result has fewer than 5 decimal digits, you may omit trailing zeros.

Output Format:
Your final answer should be a single decimal number.
Example: `2.24573` (do **NOT** include the backticks or quotes).
"""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "1/(1+|answer-gold|)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 2.0,
                 **kwargs) :
        """
        Initializes the RootExtraction_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }

        if self.rewards["rewarding_strategy"] == "1/(1+|answer-gold|)^beta" :
            self.passing_reward_threshold = rewarding_weight * ((1 / (1 + 1E-4)) ** rewarding_beta)
        else :
            raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 1, "MAX_N should be greater than or equal to 1"

        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 1, "MAX_K should be greater than or equal to 1"

        self.parameter["N"] = random.randint(1, MAX_N)
        self.parameter["K"] = random.randint(1, MAX_K)
        self.parameter["reference_answer"] = round(self.parameter["N"] ** (1 / self.parameter["K"]), 5)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"])


    def _process(self, answer : Optional[str]) -> Optional[float] :
        if answer is not None :
            answer = answer.strip()
            try :
                float_answer = float(answer)
                if not math.isfinite(float_answer) :
                    return None
                return float_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if self.rewards["rewarding_strategy"] == "1/(1+|answer-gold|)^beta" :
                return self.rewards["rewarding_weight"] * ((1 / (1 + abs(processed_result - self.parameter["reference_answer"]))) ** self.rewards["rewarding_beta"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]