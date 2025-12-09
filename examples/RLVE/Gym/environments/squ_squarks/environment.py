import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SquSquarks_Environment(VerifiableEnvironment):  # Source: https://www.luogu.com.cn/problem/P3194
    prompt_template = \
r"""Please find {N} **distinct positive integers** such that the sums of all {N} * ({N} - 1) / 2 distinct pairs among them (in any order) are exactly: {sums}
Output these {N} integers, separated by spaces."""

    def __init__(self,
                 number_multiple : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(intersection/union)^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the SquSquarks_Environment instance.
        """
        super().__init__(**kwargs)

        self.number_multiple = number_multiple
        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_beta": rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        numbers = random.sample(range(1, N * self.number_multiple + 1), N)
        self.parameter["reference_answer"] = " ".join(map(str, numbers))

        sums = self.parameter["sums"] = []
        for i, Xi in enumerate(numbers) :
            for Xj in numbers[i + 1 :] :
                sums.append(Xi + Xj)
        assert len(sums) == N * (N - 1) // 2, "sums should have exactly N * (N - 1) / 2 elements"
        random.shuffle(sums)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            sums = ", ".join(map(str, self.parameter["sums"])),
        )


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

            if len(processed_result) != self.parameter["N"] : # N integers
                return self.rewards["invalid_solution"]
            if len(set(processed_result)) != self.parameter["N"] : # distinct
                return self.rewards["invalid_solution"]
            if not all (x >= 1 for x in processed_result) : # positive integers
                return self.rewards["invalid_solution"]
            
            intersection, union = 0, 0
            gold_basket = {}
            for s in self.parameter["sums"] :
                gold_basket[s] = gold_basket.get(s, 0) + 1
                union += 1
            for i, Xi in enumerate(processed_result) :
                for Xj in processed_result[i + 1 :] :
                    s = Xi + Xj
                    if gold_basket.get(s, 0) > 0 :
                        gold_basket[s] -= 1
                        intersection += 1
                    else :
                        union += 1
            assert intersection <= union, "intersection should not exceed union"
            
            if self.rewards["rewarding_strategy"] == "(intersection/union)^beta" :
                return ((intersection / union) ** self.rewards["rewarding_beta"]) * self.rewards["rewarding_weight"]
            elif self.rewards["rewarding_strategy"] == "intersection=union" :
                return self.rewards["rewarding_weight"] * (intersection == union)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]