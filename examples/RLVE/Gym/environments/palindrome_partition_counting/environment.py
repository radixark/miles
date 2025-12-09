import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class PalindromePartitionCounting_Environment(VerifiableEnvironment) :
    prompt_template = r"""Please count the number of ways to partition the string `{S}` into (non-empty) palindromic substrings, where the number of substrings is arbitrary."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the PalindromePartitionCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        zero_probability = random.randint(1, 9) / 10
        self.parameter["S"] = S = "".join("01"[random.random() < zero_probability] for _ in range(N))


        dpF = [1] + [0] * N
        for i in range(1, N + 1) :
            for j in range(i) :
                if S[j : i] == S[j : i][:: -1] :
                    dpF[i] += dpF[j]
        self.parameter["reference_answer"] = dpF[N]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(S = self.parameter["S"])
    

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