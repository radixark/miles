import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CRT_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a system of {M} modular congruences:
{equations}

Your task is to find **any non-negative integer x** that satisfies all of the above congruences.

**Output Format:** Your output should be a **single integer x** that satisfies all the equations."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "MAX_X" in self.parameter, "MAX_X is required in parameter"
        MAX_X = self.parameter["MAX_X"]
        assert MAX_X >= 2, "MAX_X should be greater than or equal to 2"

        X = self.parameter["reference_answer"] = random.randint(2, MAX_X)

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M should be greater than or equal to 1"


        B = self.parameter["B"] = random.sample(range(2, X + 1), min(M, X - 1))
        self.parameter["X_mod_B"] = [X % b for b in B]
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            M = len(self.parameter["B"]),
            equations = "\n".join("x ≡ {} (mod {})".format(x_mod_b, b) for x_mod_b, b in zip(self.parameter["X_mod_B"], self.parameter["B"])),
        )


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
            X = processed_result
            if X < 0 :
                return self.rewards["wrong_format"]
            
            satisfied = sum(int(X % b == x_mod_b) for x_mod_b, b in zip(self.parameter["X_mod_B"], self.parameter["B"]))

            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / len(self.parameter["B"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == len(self.parameter["B"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]