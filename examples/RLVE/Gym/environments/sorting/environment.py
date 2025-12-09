import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Sorting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given the following list of numbers:
{}
Please sort them in **ascending order**.

Your final answer should be a single line containing the sorted numbers, separated by **spaces**.
For example: `1 2 3 4 5` (do **NOT** include the backticks or quotes)."""

    def __init__(self,
                 weight_multiple : int = 5,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the Sorting_Environment instance.
        """
        super().__init__(**kwargs)

        self.weight_multiple = weight_multiple
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }


    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        self.parameter["array"] = [random.randint(0, N * self.weight_multiple) for _ in range(N)]
        assert len(self.parameter["array"]) == self.parameter["N"], "array should have the same length as N"
        self.parameter["gold_answer"] = sorted(self.parameter["array"])
        assert len(self.parameter["gold_answer"]) == self.parameter["N"], "gold_answer should have the same length as N"
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(" ".join(map(str, self.parameter["array"])))


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]