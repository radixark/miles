import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MonochromeBlockCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are building a **tower of blocks** with the following rules:
- The i-th layer (from top to bottom) must contain exactly i blocks (i is from 1 to N if the tower has N layers).
- All blocks in the same layer must be of the **same color**: either black or white.
- You may use **at most {A} black blocks** and **at most {B} white blocks** in total.
- You should build a tower with the **maximum possible number of layers (N)** under these constraints.

Please compute the total number of distinct ways to build such a tower with the **maximum number of layers**.

**Output Format:** Your final answer should be a single integer — the total number of valid tower configurations that achieve the maximum number of layers."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the MonochromeBlockCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "MAX_A_B" in self.parameter, "MAX_A_B is required in parameter"
        MAX_A_B = self.parameter["MAX_A_B"]
        assert MAX_A_B >= 1, "A and B should be greater than or equal to 1"

        A = self.parameter["A"] = random.randint(1, MAX_A_B)
        B = self.parameter["B"] = random.randint(1, MAX_A_B)


        T = 0
        while ((T + 1) * (T + 2) // 2 <= A + B) :
            T += 1

        F = [0] * (A + 1)
        F[0] = 1
        for i in range(1, T + 1) :
            for j in range(A, i - 1, -1) :
                F[j] += F[j - i]

        self.parameter["reference_answer"] = sum(F[i] for i in range(max(T * (T + 1) // 2 - B, 0), A + 1))
    
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(A = self.parameter["A"], B = self.parameter["B"])
    

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