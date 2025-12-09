import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class WeightedLIS_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given two arrays `A` and `B`, each of length {N}. Their values are (indexing starts at 0):
{A}
{B}

Your task is to select a strictly increasing sequence of indices `i1, i2, ..., ik` such that:
- 0 ≤ i1 < i2 < ... < ik < {N}
- A[i1] ≤ A[i2] ≤ ... ≤ A[ik]
- Try your best to **maximize** the sum: B[i1] + B[i2] + ... + B[ik].

Output Format:
Your final answer should be a single line containing the selected indices i1, i2, ..., ik, separated by **spaces**.
Example: `0 2 3` (do **NOT** include the backticks or quotes); this means k = 3, with i1 = 0, i2 = 2, and i3 = 3.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the WeightedLIS_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }


    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        self.parameter["arrayA"] = [random.randint(0, N) for _ in range(N)]
        assert len(self.parameter["arrayA"]) == self.parameter["N"], "A should have the same length as N"
        self.parameter["arrayB"] = [random.randint(1, N) for _ in range(N)]
        assert len(self.parameter["arrayB"]) == self.parameter["N"], "B should have the same length as N"
        
        # Dynamic programming to find the maximum sum of increasing subsequence
        dpF = [0] * N
        for i in range(N) :
            dpF[i] = self.parameter["arrayB"][i]
            for j in range(i) :
                if self.parameter["arrayA"][j] <= self.parameter["arrayA"][i] :
                    dpF[i] = max(dpF[i], dpF[j] + self.parameter["arrayB"][i])
        self.parameter["gold_answer"] = max(dpF)
        assert self.parameter["gold_answer"] > 0, "gold_answer should be greater than 0"
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], A = " ".join("A[{}]={}".format(index, value) for index, value in enumerate(self.parameter["arrayA"])), B = " ".join("B[{}]={}".format(index, value) for index, value in enumerate(self.parameter["arrayB"])))


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
            
            SumB = 0
            for i in range(len(processed_result)) :
                if not (0 <= processed_result[i] < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if i > 0 and not (processed_result[i - 1] < processed_result[i]) :
                    return self.rewards["invalid_solution"]
                if i > 0 and not (self.parameter["arrayA"][processed_result[i - 1]] <= self.parameter["arrayA"][processed_result[i]]) :
                    return self.rewards["invalid_solution"]
                SumB += self.parameter["arrayB"][processed_result[i]]
            assert SumB <= self.parameter["gold_answer"], "SumB should be less than or equal to gold_answer"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((SumB / self.parameter["gold_answer"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(SumB == self.parameter["gold_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]